"""Local Whisper worker: translates every utterance to English, and (optionally)
also transcribes the original language for side-by-side captions.
Runs fully offline once the model is cached."""

import logging
import queue
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("susurro.stt")

# Scripts written without spaces between words (Chinese, Cantonese, Japanese,
# Thai, Lao, Burmese, Khmer): joining Whisper segments with " " would inject
# spurious breaks mid-sentence.
NO_SPACE_LANGS = {"zh", "yue", "ja", "th", "lo", "my", "km"}


class Transcriber:
    def __init__(self, cfg: dict, emit):
        """emit(event: dict) is called from the worker thread."""
        self.cfg = cfg
        self.emit = emit
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.model: WhisperModel | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def load(self):
        log.info("loading whisper '%s' (%s)…", self.cfg["whisper_model"], self.cfg["compute_type"])
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

    def _decode(self, audio: np.ndarray, task: str, language=None):
        # The vocabulary prompt is English text; priming a non-English
        # transcription pass with it pulls the decoder toward Latin output.
        prompt = self.cfg.get("vocabulary") or None
        if task == "transcribe" and language not in (None, "en"):
            prompt = None
        segments, info = self.model.transcribe(
            audio, task=task, language=language,
            beam_size=self.cfg["beam_size"],
            initial_prompt=prompt,
            multilingual=language is None,  # handle code-switching within an utterance
            vad_filter=True, condition_on_previous_text=False)
        out_lang = "en" if task == "translate" else (language or info.language)
        joiner = "" if out_lang in NO_SPACE_LANGS else " "
        text = joiner.join(s.text.strip() for s in segments).strip()
        return text, info

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
                text_en, info = self._decode(audio, "translate", language=forced)
                if not text_en:
                    continue
                lang = forced or info.language
                event = {
                    "type": "caption",
                    "source": source,
                    "time": t_captured,
                    "duration": round(len(audio) / 16000, 1),
                    "lang": lang,
                    "lang_prob": round(float(info.language_probability), 2),
                    "text_en": text_en,
                    "text_orig": None,
                    "latency": None,
                }
                if lang != "en" and self.cfg["show_original"]:
                    text_orig, _ = self._decode(audio, "transcribe", language=lang)
                    if text_orig and text_orig != text_en:
                        event["text_orig"] = text_orig
                event["latency"] = round(time.time() - t0, 1)
                self.emit(event)
            except Exception:
                log.exception("transcription failed")
