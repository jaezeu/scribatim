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
import re
import threading
import time
import zlib
from collections import Counter

import numpy as np

log = logging.getLogger("scribatim.stt")

# Scripts written without spaces between words (Chinese, Cantonese, Japanese,
# Thai, Lao, Burmese, Khmer): joining Whisper segments with " " would inject
# spurious breaks mid-sentence.
NO_SPACE_LANGS = {"zh", "yue", "ja", "th", "lo", "my", "km"}

# Whisper's classic hallucinations on faint/garbled audio (echo residue the
# AEC didn't fully cancel, distant murmur). Dropped only when the segment is
# also quiet — a real, direct utterance of these is much louder.
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "bye", "you", "subscribe",
    "please subscribe", "see you next time", "so",
}
HALLUCINATION_RMS = 0.02  # direct mic speech w/ AGC is typically ≥0.05


def _top_word(text: str) -> tuple[int, float]:
    """(count of the most repeated word, its share of all words)."""
    words = re.findall(r"[^\W_]+", text.lower())
    if not words:
        return 0, 0.0
    top = Counter(words).most_common(1)[0][1]
    return top, top / len(words)


def is_hallucination(text_en: str, audio: np.ndarray) -> bool:
    """True when a decoded caption is one of Whisper's known failure shapes.

    Two shapes, two gates:
      * stock phrases ("Thank you.") and moderate word repetition are only
        dropped when the audio is too quiet to be a real, direct utterance —
        the decoder invents these on breaths / echo residue / a muted mic
        ("demokrat! demokrat! demokrat!" on silence is the classic loop);
      * an extreme loop (one word ≥5×, or a long phrase that zlib squashes
        >2.4× — Whisper's own repetition metric) is dropped at any volume:
        it is a stuck decoder, not something a person said.

    `audio` must be the raw captured segment (pre-normalization) so the RMS
    gate reflects what the microphone actually heard.
    """
    quiet = float(np.sqrt(np.mean(audio ** 2))) < HALLUCINATION_RMS
    if quiet and text_en.strip(" .!?,¡¿").lower() in HALLUCINATION_PHRASES:
        return True
    top, share = _top_word(text_en)
    if quiet and top >= 3 and share >= 0.7:
        return True
    if top >= 5 and share >= 0.8:
        return True
    raw = text_en.encode()
    return len(raw) >= 60 and len(raw) > 2.4 * len(zlib.compress(raw))


def _suspect_segment(avg_logprob: float, no_speech_prob: float,
                     compression_ratio: float, strict: bool = False) -> bool:
    """Whisper's own per-segment quality signals, applied as a drop filter:
    text decoded over what the model itself scored as silence, or text whose
    compression ratio says the decoder looped. Both backends expose these;
    neither uses them to *drop* (only for temperature fallback), so confident
    garbage still reaches the caller without this.

    `strict` is for segments the VAD already distrusts (little/no speech,
    or near-silent audio): there Whisper fabricates fluent full sentences
    with only mediocre confidence, so any single weak signal is enough to
    drop — the joint rule below only catches the blatant cases."""
    if strict:
        return (no_speech_prob > 0.5 or avg_logprob < -0.85
                or compression_ratio > 2.2)
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    return compression_ratio > 2.4


def normalize_segment(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize a quiet segment before decoding. Whisper's accuracy
    drops on faint audio (a distant speaker on the tap, an AGC-less mic);
    the gain is capped so near-silence isn't amplified into decodable noise,
    and already-loud audio is left untouched."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-4:
        return audio
    gain = min(0.9 / peak, 12.0)
    if gain <= 1.0:
        return audio
    return (audio * gain).astype(np.float32)

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

    def drain(self, timeout: float = 60.0) -> bool:
        """Block until every queued segment has been transcribed (captions
        keep emitting meanwhile). The last utterances of a meeting are still
        in this queue when capture stops — the saved transcript must wait
        for them. Returns False if the timeout expired first."""
        deadline = time.time() + timeout
        with self.q.all_tasks_done:
            while self.q.unfinished_tasks:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self.q.all_tasks_done.wait(remaining)
        return True

    def submit(self, source: str, audio: np.ndarray):
        try:
            self.q.put_nowait((source, audio, time.time()))
        except queue.Full:
            log.warning("transcription queue full, dropping %.1fs segment", len(audio) / 16000)

    def _speech_ratio(self, audio: np.ndarray) -> float:
        """Fraction of the segment Silero VAD scores as speech. Whisper
        hallucinates captions (stock phrases, fluent invented sentences) on
        breaths/keyboard/noise that pass the energy gate; the ratio decides
        whether a segment is skipped outright (0.0) or decoded in strict
        mode (low ratio). Must be fed the RAW capture — the pre-decode gain
        boost makes room noise look more speech-like to the VAD."""
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            spans = get_speech_timestamps(
                audio, VadOptions(threshold=0.5, min_speech_duration_ms=250))
            return sum(s["end"] - s["start"] for s in spans) / len(audio)
        except Exception:
            return 1.0  # never drop captions because the VAD failed

    def _prompt_for(self, task: str, language):
        # The vocabulary prompt is English text; priming a non-English
        # transcription pass with it pulls the decoder toward Latin output.
        prompt = self.cfg.get("vocabulary") or None
        if task == "transcribe" and language not in (None, "en"):
            prompt = None
        return prompt

    def _decode(self, audio: np.ndarray, task: str, language=None, strict=False):
        """Returns (text, detected_language, language_probability)."""
        if self._mlx_repo:
            return self._decode_mlx(audio, task, language, strict)
        return self._decode_ct2(audio, task, language, strict)

    def _decode_mlx(self, audio: np.ndarray, task: str, language=None, strict=False):
        result = self._mlx.transcribe(
            audio, path_or_hf_repo=self._mlx_repo,
            task=task, language=language,
            initial_prompt=self._prompt_for(task, language),
            condition_on_previous_text=False, verbose=None)
        # Concatenating segment texts verbatim reproduces result["text"]
        # (each segment carries its own leading space where the script wants
        # one), so filtering low-quality segments keeps the join script-aware.
        segs = [s for s in result.get("segments", [])
                if not _suspect_segment(s.get("avg_logprob", 0.0),
                                        s.get("no_speech_prob", 0.0),
                                        s.get("compression_ratio", 0.0), strict)]
        lang = language or result.get("language") or "en"
        return "".join(s["text"] for s in segs).strip(), lang, 1.0

    def _decode_ct2(self, audio: np.ndarray, task: str, language=None, strict=False):
        segments, info = self.model.transcribe(
            audio, task=task, language=language,
            beam_size=self.cfg["beam_size"],
            initial_prompt=self._prompt_for(task, language),
            multilingual=language is None,  # handle code-switching within an utterance
            vad_filter=True, condition_on_previous_text=False)
        out_lang = "en" if task == "translate" else (language or info.language)
        joiner = "" if out_lang in NO_SPACE_LANGS else " "
        text = joiner.join(
            s.text.strip() for s in segments
            if not _suspect_segment(s.avg_logprob, s.no_speech_prob,
                                    s.compression_ratio, strict)).strip()
        return text, info.language, float(info.language_probability)

    def _run(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            source, audio, t_captured = item
            try:
                t0 = time.time()
                # Optional language lock: per-utterance auto-detection is
                # unreliable on short clips with smaller models (e.g. zh/yue/ja
                # confusion) — cfg["language"] pins the source language instead.
                forced = self.cfg.get("language") or None
                # VAD on the RAW audio: no speech at all → skip; marginal
                # speech or near-silence → strict mode (no gain boost, and
                # any single weak quality signal drops a decoded segment).
                # Whisper writes fluent fiction over exactly these segments.
                ratio = self._speech_ratio(audio)
                if ratio <= 0.0:
                    log.info("dropped %.1fs segment: VAD found no speech",
                             len(audio) / 16000)
                    continue
                raw_rms = float(np.sqrt(np.mean(audio ** 2)))
                strict = ratio < 0.4 or raw_rms < HALLUCINATION_RMS
                norm = audio if strict else normalize_segment(audio)
                text_en, detected, prob = self._decode(
                    norm, "translate", language=forced, strict=strict)
                if not text_en:
                    continue
                if is_hallucination(text_en, audio):
                    log.info("dropped likely hallucination: %r", text_en)
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
                    text_orig, _, _ = self._decode(
                        norm, "transcribe", language=lang, strict=strict)
                    if text_orig and text_orig != text_en:
                        event["text_orig"] = text_orig
                    elif text_orig == text_en:
                        # a real non-English utterance transcribes differently
                        # than it translates; verbatim agreement means the
                        # short-clip language detection misfired on English
                        event["lang"] = "en"
                event["latency"] = round(time.time() - t0, 1)
                self.emit(event)
            except Exception:
                log.exception("transcription failed")
            finally:
                self.q.task_done()
