"""Local Whisper worker: translates every utterance to English, and (optionally)
also transcribes the original language for side-by-side captions.
Runs fully offline once the model is cached.

Two backends, picked automatically:
  * MLX (Apple Silicon) — Whisper on the M-series GPU via Metal; typically
    several times faster than CPU, which is what lets `medium`/`large-v3`
    keep up with live speech.
  * CTranslate2 / faster-whisper (CPU) — Intel Macs, or fallback whenever
    MLX is unavailable.
"""

import logging
import platform
import queue
import threading
import time

import numpy as np

log = logging.getLogger("susurro.stt")

# Scripts written without spaces between words (Chinese, Cantonese, Japanese,
# Thai, Lao, Burmese, Khmer): joining Whisper segments with " " would inject
# spurious breaks mid-sentence.
NO_SPACE_LANGS = {"zh", "yue", "ja", "th", "lo", "my", "km"}

# Verified mlx-community conversions of the OpenAI Whisper weights.
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class Transcriber:
    def __init__(self, cfg: dict, emit):
        """emit(event: dict) is called from the worker thread."""
        self.cfg = cfg
        self.emit = emit
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.model = None            # CTranslate2 WhisperModel when on CPU
        self._mlx = None             # mlx_whisper module when on Metal
        self._mlx_repo: str | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def load(self):
        backend = self.cfg.get("whisper_backend", "auto")
        if backend in ("auto", "mlx"):
            if self._load_mlx(required=backend == "mlx"):
                return
        self._load_ct2()

    def _load_mlx(self, required: bool) -> bool:
        model = self.cfg["whisper_model"]
        repo = model if "/" in model else MLX_REPOS.get(model)
        try:
            if platform.machine() != "arm64":
                raise RuntimeError("MLX needs Apple Silicon")
            if not repo:
                raise RuntimeError(f"no known MLX conversion of '{model}'")
            import mlx_whisper
            log.info("loading whisper '%s' on Metal GPU (MLX)…", repo)
            t0 = time.time()
            # warm-up on silence: downloads the weights once, then caches
            # the loaded model inside mlx_whisper for subsequent calls
            mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32), path_or_hf_repo=repo, verbose=None)
            self._mlx, self._mlx_repo = mlx_whisper, repo
            log.info("whisper (MLX) ready in %.1fs", time.time() - t0)
            return True
        except Exception as e:
            if required:
                raise
            log.warning("MLX backend unavailable (%s) — using CPU", e)
            return False

    def _load_ct2(self):
        from faster_whisper import WhisperModel
        log.info("loading whisper '%s' on CPU (%s)…",
                 self.cfg["whisper_model"], self.cfg["compute_type"])
        t0 = time.time()
        try:
            # cached weights: stay fully offline, no HuggingFace revision checks
            self.model = WhisperModel(
                self.cfg["whisper_model"], device="cpu",
                compute_type=self.cfg["compute_type"], local_files_only=True)
        except Exception:
            log.info("model not cached yet — downloading once")
            self.model = WhisperModel(
                self.cfg["whisper_model"], device="cpu",
                compute_type=self.cfg["compute_type"])
        log.info("whisper ready in %.1fs", time.time() - t0)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.q.put(None)

    def submit(self, source: str, audio: np.ndarray):
        try:
            self.q.put_nowait((source, audio, time.time()))
        except queue.Full:
            log.warning("transcription queue full, dropping %.1fs segment", len(audio) / 16000)

    def _prompt_for(self, task: str, language):
        # The vocabulary prompt is English text; priming a non-English
        # transcription pass with it pulls the decoder toward Latin output.
        prompt = self.cfg.get("vocabulary") or None
        if task == "transcribe" and language not in (None, "en"):
            prompt = None
        return prompt

    def _decode(self, audio: np.ndarray, task: str, language=None):
        """Returns (text, detected_language, language_probability)."""
        if self._mlx_repo:
            return self._decode_mlx(audio, task, language)
        return self._decode_ct2(audio, task, language)

    def _decode_mlx(self, audio: np.ndarray, task: str, language=None):
        result = self._mlx.transcribe(
            audio, path_or_hf_repo=self._mlx_repo,
            task=task, language=language,
            initial_prompt=self._prompt_for(task, language),
            condition_on_previous_text=False, verbose=None)
        # Whisper's own segment concatenation is script-aware, so no joiner
        # fix-up is needed here.
        lang = language or result.get("language") or "en"
        return result["text"].strip(), lang, 1.0

    def _decode_ct2(self, audio: np.ndarray, task: str, language=None):
        segments, info = self.model.transcribe(
            audio, task=task, language=language,
            beam_size=self.cfg["beam_size"],
            initial_prompt=self._prompt_for(task, language),
            multilingual=language is None,  # handle code-switching within an utterance
            vad_filter=True, condition_on_previous_text=False)
        out_lang = "en" if task == "translate" else (language or info.language)
        joiner = "" if out_lang in NO_SPACE_LANGS else " "
        text = joiner.join(s.text.strip() for s in segments).strip()
        return text, info.language, float(info.language_probability)

    def _run(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                break
            source, audio, t_captured = item
            try:
                t0 = time.time()
                # Optional language lock: per-utterance auto-detection is
                # unreliable on short clips with smaller models (e.g. zh/yue/ja
                # confusion) — cfg["language"] pins the source language instead.
                forced = self.cfg.get("language") or None
                text_en, detected, prob = self._decode(audio, "translate", language=forced)
                if not text_en:
                    continue
                lang = forced or detected
                event = {
                    "type": "caption",
                    "source": source,
                    "time": t_captured,
                    "duration": round(len(audio) / 16000, 1),
                    "lang": lang,
                    "lang_prob": round(prob, 2),
                    "text_en": text_en,
                    "text_orig": None,
                    "latency": None,
                }
                if lang != "en" and self.cfg["show_original"]:
                    text_orig, _, _ = self._decode(audio, "transcribe", language=lang)
                    if text_orig and text_orig != text_en:
                        event["text_orig"] = text_orig
                event["latency"] = round(time.time() - t0, 1)
                self.emit(event)
            except Exception:
                log.exception("transcription failed")
