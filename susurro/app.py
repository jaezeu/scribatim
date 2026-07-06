"""Susurro server: orchestrates capture → whisper → live dashboard → minutes.

Security posture:
  * binds 127.0.0.1 only — unreachable from the network
  * every request must carry a per-launch random token (URL once, then cookie)
  * no external calls at runtime; audio is never written to disk
  * transcripts/minutes saved locally under ~/Documents/Susurro (0700)
"""

import asyncio
import contextlib
import json
import logging
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .audio import MicSource, Segmenter, SystemAudioSource
from .config import PROJECT_ROOT, load_config
from .minutes import PROMPTS, generate, transcript_to_text
from .speaker import SpeakerTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("susurro")

cfg = load_config()
TOKEN = secrets.token_urlsafe(16)


def _warmup_audio():
    # Initialize PortAudio and enumerate devices before any capture exists:
    # doing this while the tap's aggregate device appears can wedge CoreAudio.
    import sounddevice as sd
    sd.query_devices()


@contextlib.asynccontextmanager
async def lifespan(app):
    state.loop = asyncio.get_running_loop()
    from .transcriber import Transcriber
    state.transcriber = Transcriber(cfg, broadcast)
    await asyncio.to_thread(_warmup_audio)
    await asyncio.to_thread(state.transcriber.load)
    state.transcriber.start()
    log.info("dashboard ready → http://127.0.0.1:%d/?t=%s", cfg["port"], TOKEN)
    yield
    state.transcriber.stop()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

ALLOWED_HOSTS = {f"127.0.0.1:{cfg['port']}", f"localhost:{cfg['port']}"}


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.started_at: float | None = None
        self.captions: list[dict] = []
        self.subscribers: list[asyncio.Queue] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sources = []
        self.transcriber = None
        self.deliverables: dict[str, str] = {}  # kind -> markdown
        self.session_dir: Path | None = None
        self.speaker: SpeakerTracker | None = None
        self.speaker_enabled: bool = False


state = State()
state.speaker_enabled = bool(cfg["speaker_ocr"])


def broadcast(event: dict):
    # experimental speaker OCR: label system-audio captions with the name
    # shown on the meeting window while that segment was being spoken
    if event["type"] == "caption" and event["source"] == "system" and state.speaker:
        event["speaker"] = state.speaker.name_for(
            event["time"] - event["duration"], event["time"])
    if event["type"] == "caption":
        with state.lock:
            state.captions.append(event)
    if state.loop:
        for q in list(state.subscribers):
            state.loop.call_soon_threadsafe(q.put_nowait, event)


def _start_speaker_tracker() -> str | None:
    """Start speaker OCR if enabled; returns a warning string on failure."""
    if not (state.speaker_enabled and state.speaker is None):
        return None
    try:
        tracker = SpeakerTracker()
        tracker.start()
        state.speaker = tracker
        return None
    except Exception as e:
        log.warning("speaker OCR unavailable: %s", e)
        return f"speaker names: {e}"


def _stop_speaker_tracker():
    tracker, state.speaker = state.speaker, None
    if tracker:
        tracker.stop()


@app.middleware("http")
async def auth(request: Request, call_next):
    # DNS-rebinding defense: only accept the loopback hostnames we serve on
    if request.headers.get("host", "") not in ALLOWED_HOSTS:
        return JSONResponse({"error": "bad host"}, status_code=403)
    supplied = request.query_params.get("t") or request.cookies.get("susurro_token")
    if not (supplied and secrets.compare_digest(supplied, TOKEN)):
        return JSONResponse({"error": "missing or bad token"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
    if request.query_params.get("t"):
        response.set_cookie("susurro_token", TOKEN, httponly=True, samesite="strict")
    return response


@app.get("/")
async def index():
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/api/status")
async def status():
    with state.lock:
        return {
            "running": state.running,
            "started_at": state.started_at,
            "captions": len(state.captions),
            "model": cfg["whisper_model"],
            "ollama_model": cfg["ollama_model"],
            "deliverables": sorted(state.deliverables),
        }


@app.post("/api/start")
async def start():
    with state.lock:
        if state.running:
            return {"ok": True, "already": True}
        state.running = True
        state.started_at = time.time()
        state.captions = []
        state.deliverables = {}
        state.session_dir = None

    def on_segment(source, audio):
        state.transcriber.submit(source, audio)

    def on_level(source, rms):
        broadcast({"type": "level", "source": source, "rms": round(rms, 4)})

    seg_kwargs = dict(
        silence_s=cfg["segment_silence_seconds"],
        max_s=cfg["segment_max_seconds"],
        min_speech_s=cfg["segment_min_speech_seconds"])

    errors = []
    system = SystemAudioSource(Segmenter("system", on_segment, on_level, **seg_kwargs))
    mic = MicSource(Segmenter("mic", on_segment, on_level, **seg_kwargs),
                    aec=cfg["mic_aec"])
    # mic first: opening it during the tap's aggregate-device creation can hang
    for name, src in (("mic", mic), ("system", system)):
        try:
            # keep the meeting usable even if one source wedges (e.g. a
            # pending permission prompt): give it 10s, then move on
            await asyncio.wait_for(asyncio.to_thread(src.start), timeout=10)
            state.sources.append(src)
        except asyncio.TimeoutError:
            log.error("%s source did not start within 10s", name)
            errors.append(f"{name}: did not start within 10s (permission prompt pending?)")
            state.sources.append(src)  # so /api/stop still cleans it up if it comes up late
        except Exception as e:
            log.exception("failed to start %s source", name)
            errors.append(f"{name}: {e}")

    if len(errors) == 2:  # neither source came up
        await stop()
        return JSONResponse({"ok": False, "errors": errors}, status_code=500)

    warning = await asyncio.to_thread(_start_speaker_tracker)
    if warning:
        errors.append(warning)  # non-fatal: captions just lack names

    broadcast({"type": "status", "running": True, "started_at": state.started_at,
               "errors": errors})
    return {"ok": True, "errors": errors}


@app.post("/api/stop")
async def stop():
    with state.lock:
        if not state.running:
            return {"ok": True, "already": True}
        state.running = False
        sources, state.sources = state.sources, []
    for src in sources:
        try:
            await asyncio.to_thread(src.stop)
        except Exception:
            log.exception("source stop failed")
    await asyncio.to_thread(_stop_speaker_tracker)
    _save_session()
    broadcast({"type": "status", "running": False, "started_at": None, "errors": []})
    return {"ok": True, "captions": len(state.captions)}


@app.post("/api/speaker/{action}")
async def speaker_toggle(action: str):
    if action not in ("on", "off"):
        return JSONResponse({"ok": False, "error": "unknown action"}, status_code=404)
    state.speaker_enabled = action == "on"
    if not state.speaker_enabled:
        await asyncio.to_thread(_stop_speaker_tracker)
    elif state.running:
        warning = await asyncio.to_thread(_start_speaker_tracker)
        if warning:
            state.speaker_enabled = False
            broadcast({"type": "speaker", "enabled": False})
            return JSONResponse({"ok": False, "error": warning}, status_code=503)
    broadcast({"type": "speaker", "enabled": state.speaker_enabled})
    return {"ok": True, "enabled": state.speaker_enabled}


@app.post("/api/generate/{kind}")
async def generate_deliverable(kind: str):
    if kind not in PROMPTS:
        return JSONResponse({"ok": False, "error": "unknown deliverable"}, status_code=404)
    with state.lock:
        captions = list(state.captions)
    try:
        md = await asyncio.to_thread(generate, cfg, kind, captions)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)
    with state.lock:
        state.deliverables[kind] = md
    path = _save_session()
    return {"ok": True, "kind": kind, "markdown": md,
            "saved_to": str(path) if path else None}


@app.get("/api/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    state.subscribers.append(q)

    async def stream():
        try:
            with state.lock:
                snapshot = {
                    "type": "snapshot",
                    "running": state.running,
                    "started_at": state.started_at,
                    "captions": state.captions[-200:],
                    "model": cfg["whisper_model"],
                    "ollama_model": cfg["ollama_model"],
                    "speaker_enabled": state.speaker_enabled,
                }
            yield f"data: {json.dumps(snapshot)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    break
        finally:
            state.subscribers.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


def _save_session() -> Path | None:
    with state.lock:
        captions = list(state.captions)
        deliverables = dict(state.deliverables)
        started = state.started_at or time.time()
        if not captions:
            return None
        if state.session_dir is None:
            root = Path(cfg["save_dir"])
            root.mkdir(parents=True, exist_ok=True)
            root.chmod(0o700)
            state.session_dir = root / time.strftime("%Y-%m-%d_%H%M", time.localtime(started))
            state.session_dir.mkdir(exist_ok=True)
            state.session_dir.chmod(0o700)
        session_dir = state.session_dir

    (session_dir / "transcript.md").write_text(
        "# Transcript — " + time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
        + "\n\n```\n" + transcript_to_text(captions) + "\n```\n")
    for kind, md in deliverables.items():
        (session_dir / f"{kind}.md").write_text(md + "\n")
    (session_dir / "meta.json").write_text(json.dumps({
        "started_at": started, "captions": len(captions),
        "whisper_model": cfg["whisper_model"], "ollama_model": cfg["ollama_model"],
    }, indent=2))
    log.info("session saved to %s", session_dir)
    return session_dir


def main():
    import uvicorn
    print(f"\n  Susurro → http://127.0.0.1:{cfg['port']}/?t={TOKEN}\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=cfg["port"], log_level="warning")


if __name__ == "__main__":
    main()
