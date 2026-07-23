"""Audio capture: system-output tap (remote participants) + microphone (you).

Both sources deliver mono float32 @ 16 kHz to a Segmenter, which uses an
adaptive energy gate to cut speech into utterance-sized segments for Whisper.
On open speakers the mic also hears the remote participants; EchoGate drops
those mic segments by correlating them against the system tap.
"""

import json
import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger("scribatim.audio")

TARGET_RATE = 16000
BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
TAP_BINARY = BIN_DIR / "scribatim-tap"


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
                 silence_s=0.8, max_s=12.0, min_speech_s=0.4, tee=None):
        self.source = source
        self.on_segment = on_segment          # (source, np.ndarray) -> None
        self.on_level = on_level              # (source, float rms) -> None
        self.tee = tee                        # (np.ndarray) -> None, raw 16 kHz stream
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
        if self.tee is not None:
            self.tee(chunk)
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


class MicSource:
    """Raw microphone capture from the default input device — the built-in
    mic, or whatever headset is selected in System Settings.

    Deliberately does NOT use Apple's voice-processing unit (AEC): macOS
    applies that processing device-wide, so while it is engaged every other
    client of the microphone — Teams, Zoom — receives a signal attenuated by
    ~40 dB and the user becomes inaudible in their own meeting (measured: a
    concurrent raw capture drops to ~1% amplitude the moment the voice unit
    engages, and recovers when it disengages). Speaker bleed of remote
    voices into an open mic is instead handled downstream by EchoGate,
    which touches nothing system-wide.
    """

    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter
        self.stream = None

    def start(self):
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
        try:
            device = sd.query_devices(kind="input")["name"]
        except Exception:
            device = "default input"
        log.info("microphone streaming (%s)", device)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.segmenter.flush()


class EchoGate:
    """Drops mic segments that are speaker bleed of the system output.

    On open speakers the microphone hears the remote participants; those
    utterances are already transcribed from the system tap, so decoding them
    again from the mic lane duplicates every caption attributed to "You".
    The tap is a clean reference of exactly what the speakers played: a mic
    segment that is essentially a delayed, room-filtered copy of it peaks
    high in normalized cross-correlation against the time-aligned reference
    window; the user's own voice does not. On a headset there is no acoustic
    path from the speakers to the mic, so the gate never fires.

    Timeline: reference chunks arrive as a contiguous 16 kHz stream whose
    latest sample is stamped with wall-clock time; a mic segment ends at the
    moment its Segmenter closes it. Pipe buffering skews the two clocks by a
    fraction of a second, so correlation is searched over ±PAD_S of lag.
    """

    KEEP_S = 60.0        # reference history kept — the echo check runs on the
                         # transcriber worker, so it must still cover a segment
                         # that sat behind a queue backlog
    PAD_S = 0.5          # timestamp jitter absorbed by the lag search
    MIN_REF_RMS = 0.004  # quieter than this = speakers effectively silent
    THRESHOLD = 0.30     # unrelated speech correlates ~0.05; bleed ~0.4-0.9

    def __init__(self):
        self._chunks: deque = deque()
        self._size = 0          # samples currently held
        self._end = 0.0         # wall-clock time of the last reference sample
        self._lock = threading.Lock()

    def feed_reference(self, chunk: np.ndarray, now: float | None = None):
        with self._lock:
            self._chunks.append(chunk)
            self._size += len(chunk)
            self._end = time.time() if now is None else now
            keep = int(self.KEEP_S * TARGET_RATE)
            while self._size - len(self._chunks[0]) > keep:
                self._size -= len(self._chunks.popleft())

    def _reference(self, a: float, b: float) -> np.ndarray | None:
        """Reference samples covering wall-clock [a, b], or None if absent."""
        with self._lock:
            if not self._chunks:
                return None
            buf = np.concatenate(self._chunks)
            end = self._end
        start = end - len(buf) / TARGET_RATE
        i0 = max(int((a - start) * TARGET_RATE), 0)
        i1 = min(int((b - start) * TARGET_RATE), len(buf))
        return buf[i0:i1] if i1 > i0 else None

    def is_echo(self, segment: np.ndarray, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        t0 = now - len(segment) / TARGET_RATE
        ref = self._reference(t0 - self.PAD_S, now + self.PAD_S)
        # need at least some lag slack beyond the segment length to align
        if ref is None or len(ref) < len(segment) + TARGET_RATE // 8:
            return False
        if float(np.sqrt(np.mean(ref ** 2))) < self.MIN_REF_RMS:
            return False  # nothing loud was playing — can't be bleed
        corr = self._max_ncc(ref, segment)
        if corr >= self.THRESHOLD:
            log.info("mic segment dropped as speaker bleed (corr %.2f)", corr)
            return True
        return False

    @staticmethod
    def _max_ncc(ref: np.ndarray, seg: np.ndarray) -> float:
        """Peak normalized cross-correlation of seg against every lag of ref."""
        seg = seg - float(np.mean(seg))
        ref = ref - float(np.mean(ref))
        seg_norm = float(np.sqrt(np.sum(seg.astype(np.float64) ** 2)))
        if seg_norm < 1e-6:
            return 0.0
        nlag = len(ref) - len(seg)
        nfft = 1 << int(len(ref) + len(seg)).bit_length()
        r = np.fft.rfft(ref, nfft)
        s = np.fft.rfft(seg, nfft)
        corr = np.fft.irfft(r * np.conj(s), nfft)[:nlag + 1]
        csum = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
        window_energy = csum[len(seg):len(seg) + nlag + 1] - csum[:nlag + 1]
        ncc = np.abs(corr) / (np.sqrt(np.maximum(window_energy, 1e-12)) * seg_norm)
        return float(np.max(ncc))
