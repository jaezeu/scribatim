"""Audio capture: system-output tap (remote participants) + microphone (you).

Both sources deliver mono float32 @ 16 kHz to a Segmenter, which uses an
adaptive energy gate to cut speech into utterance-sized segments for Whisper.
"""

import json
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("scribatim.audio")

TARGET_RATE = 16000
BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
TAP_BINARY = BIN_DIR / "scribatim-tap"
MIC_BINARY = BIN_DIR / "scribatim-mic"


class Resampler:
    """Streaming downsampler to 16 kHz with a proper anti-aliasing low-pass.

    The naive alternatives — box-averaging 48 kHz (integer ratio) or bare
    linear interpolation of the mic helper's 24 kHz — fold everything above
    8 kHz back into the speech band; that aliasing is inaudible in a meter
    but measurably hurts Whisper. A short windowed-sinc FIR (cutoff 7.2 kHz)
    ahead of the rate conversion removes it for ~0.1 ms of CPU per second of
    audio. Stateful: chunk boundaries are seamless, so it can sit directly
    behind the helper pipe reads.
    """

    TAPS = 63  # ~2 ms group delay at 24/48 kHz — irrelevant for captions

    def __init__(self, rate: int):
        self.rate = rate
        self.step = rate / TARGET_RATE  # source samples per output sample
        fc = min(0.45 * TARGET_RATE, 0.45 * rate) / rate  # cycles/sample
        k = np.arange(self.TAPS) - (self.TAPS - 1) / 2
        kernel = 2 * fc * np.sinc(2 * fc * k) * np.hamming(self.TAPS)
        self._kernel = (kernel / kernel.sum()).astype(np.float32)
        self._tail = np.zeros(self.TAPS - 1, dtype=np.float32)  # filter history
        self._carry = np.empty(0, dtype=np.float32)  # filtered, not yet consumed
        self._pos = 0.0  # next output position within _carry+filtered, in source samples

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        if self.rate == TARGET_RATE:
            return chunk
        buf = np.concatenate([self._tail, chunk.astype(np.float32, copy=False)])
        if len(buf) < self.TAPS:
            self._tail = buf
            return np.empty(0, dtype=np.float32)
        filtered = np.convolve(buf, self._kernel, mode="valid").astype(np.float32)
        self._tail = buf[-(self.TAPS - 1):]
        stream = np.concatenate([self._carry, filtered])
        count = int((len(stream) - 1 - self._pos) // self.step) + 1
        if count <= 0:
            self._carry = stream
            return np.empty(0, dtype=np.float32)
        positions = self._pos + self.step * np.arange(count)
        out = np.interp(positions, np.arange(len(stream)), stream).astype(np.float32)
        next_pos = self._pos + self.step * count
        keep_from = min(int(next_pos), len(stream) - 1)
        self._carry = stream[keep_from:]
        self._pos = next_pos - keep_from
        return out


class Segmenter:
    """Adaptive energy-gated utterance segmentation on a 16 kHz mono stream."""

    FRAME = int(TARGET_RATE * 0.03)  # 30 ms

    def __init__(self, source: str, on_segment, on_level,
                 silence_s=0.8, max_s=12.0, min_speech_s=0.4):
        self.source = source
        self.on_segment = on_segment          # (source, np.ndarray) -> None
        self.on_level = on_level              # (source, float rms) -> None
        self.silence_frames = int(silence_s / 0.03)
        self.max_frames = int(max_s / 0.03)
        self.min_speech_frames = int(min_speech_s / 0.03)
        self.noise_floor = 0.003
        self.pre_roll: list[np.ndarray] = []  # ~0.3 s kept before speech onset
        self.frames: list[np.ndarray] = []
        self.speech_count = 0
        self.trailing_silence = 0
        self.in_speech = False
        self._residual = np.empty(0, dtype=np.float32)
        self._level_acc = 0.0
        self._level_n = 0

    def feed(self, chunk: np.ndarray):
        buf = np.concatenate([self._residual, chunk])
        n_frames = len(buf) // self.FRAME
        self._residual = buf[n_frames * self.FRAME:]
        for i in range(n_frames):
            self._frame(buf[i * self.FRAME:(i + 1) * self.FRAME])

    def _frame(self, frame: np.ndarray):
        rms = float(np.sqrt(np.mean(frame ** 2)))

        self._level_acc = max(self._level_acc, rms)
        self._level_n += 1
        if self._level_n >= 7:  # ~200 ms
            self.on_level(self.source, self._level_acc)
            self._level_acc, self._level_n = 0.0, 0

        threshold = max(self.noise_floor * 3.0, 0.006)
        is_speech = rms > threshold
        if not is_speech:
            # slowly track the noise floor on quiet frames
            self.noise_floor = 0.97 * self.noise_floor + 0.03 * max(rms, 1e-5)

        if not self.in_speech:
            self.pre_roll.append(frame)
            if len(self.pre_roll) > 10:  # 0.3 s
                self.pre_roll.pop(0)
            if is_speech:
                self.in_speech = True
                self.frames = list(self.pre_roll)
                self.pre_roll = []
                self.speech_count = 1
                self.trailing_silence = 0
            return

        self.frames.append(frame)
        if is_speech:
            self.speech_count += 1
            self.trailing_silence = 0
        else:
            self.trailing_silence += 1

        if self.trailing_silence >= self.silence_frames or len(self.frames) >= self.max_frames:
            self._close()

    def _close(self):
        frames, speech = self.frames, self.speech_count
        self.frames, self.speech_count = [], 0
        self.in_speech, self.trailing_silence = False, 0
        if speech >= self.min_speech_frames:
            self.on_segment(self.source, np.concatenate(frames))

    def flush(self):
        if self.in_speech and self.frames:
            self._close()


class HelperProcessSource:
    """Spawns a Swift capture helper and streams its float32 audio.

    Helper protocol: one JSON header line {"rate": N, "channels": 1} on
    stdout, then raw little-endian float32 mono samples.
    """

    binary: Path
    label = "helper"

    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter
        self.proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    def start(self):
        if not self.binary.exists():
            raise RuntimeError(f"{self.label} binary missing: {self.binary} — run setup.sh")
        self.proc = subprocess.Popen(
            [str(self.binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        line = self.proc.stdout.readline()
        if not line:  # helper died before the header — surface its stderr
            err = self.proc.stderr.read().decode(errors="replace").strip()
            self.stop()
            raise RuntimeError(f"{self.label} failed to start: {err or 'no output'}")
        header = json.loads(line)
        self.rate = int(header["rate"])
        log.info("%s streaming at %d Hz", self.label, self.rate)
        for target in (self._read_audio, self._read_stderr):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def _read_audio(self):
        assert self.proc and self.proc.stdout
        resampler = Resampler(self.rate)
        while True:
            data = self.proc.stdout.read(self.rate)  # rate/4 frames ≈ 0.25 s per read
            if not data:
                break
            samples = np.frombuffer(data[:len(data) - len(data) % 4], dtype="<f4")
            if len(samples):
                out = resampler.feed(samples)
                if len(out):
                    self.segmenter.feed(out)
        self.segmenter.flush()

    def _read_stderr(self):
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            log.info("%s: %s", self.label, line.decode(errors="replace").rstrip())

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


class SystemAudioSource(HelperProcessSource):
    """System-output audio (remote participants) via the Core Audio tap."""

    binary = TAP_BINARY
    label = "system tap"


class AECMicSource(HelperProcessSource):
    """Microphone through Apple's voice-processing unit: echo cancellation
    (system playback subtracted from the mic), noise suppression, auto gain.
    Keeps the mic lane clean even on open speakers."""

    binary = MIC_BINARY
    label = "mic (echo-cancelled)"


class MicSource:
    """Microphone capture. Prefers the echo-cancelled Swift helper; falls back
    to a raw sounddevice stream if the helper is missing or fails."""

    def __init__(self, segmenter: Segmenter, aec: bool = True):
        self.segmenter = segmenter
        self.aec = aec
        self.stream = None
        self._helper: AECMicSource | None = None

    def start(self):
        if self.aec and MIC_BINARY.exists():
            try:
                self._helper = AECMicSource(self.segmenter)
                self._helper.start()
                return
            except Exception as e:
                log.warning("echo-cancelled mic unavailable (%s) — using raw mic", e)
                self._helper = None
        self._start_raw()

    def _start_raw(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                log.warning("mic status: %s", status)
            self.segmenter.feed(indata[:, 0].copy())

        try:
            self.stream = sd.InputStream(
                samplerate=TARGET_RATE, channels=1, dtype="float32",
                blocksize=int(TARGET_RATE * 0.1), callback=callback)
            self.stream.start()
        except Exception:
            # device refuses 16 kHz: open at native rate and resample
            info = sd.query_devices(kind="input")
            native = int(info["default_samplerate"])
            log.info("mic falling back to native %d Hz", native)
            resampler = Resampler(native)

            def cb_native(indata, frames, time_info, status):
                out = resampler.feed(indata[:, 0].copy())
                if len(out):
                    self.segmenter.feed(out)

            self.stream = sd.InputStream(
                samplerate=native, channels=1, dtype="float32",
                blocksize=int(native * 0.1), callback=cb_native)
            self.stream.start()
        log.info("microphone streaming")

    def stop(self):
        if self._helper:
            self._helper.stop()
            self._helper = None
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.segmenter.flush()
