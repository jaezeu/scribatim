"""Tests for the streaming anti-aliased resampler and the echo gate."""

import unittest

import numpy as np

from scribatim.audio import TARGET_RATE, EchoGate, Resampler, Segmenter


def sine(freq, rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


class ResamplerTest(unittest.TestCase):
    def test_output_length_matches_ratio_24k(self):
        out = Resampler(24000).feed(sine(440, 24000))
        self.assertAlmostEqual(len(out), TARGET_RATE, delta=Resampler.TAPS)

    def test_output_length_matches_ratio_48k(self):
        out = Resampler(48000).feed(sine(440, 48000))
        self.assertAlmostEqual(len(out), TARGET_RATE, delta=Resampler.TAPS)

    def test_speech_band_tone_preserved(self):
        # 1 kHz is squarely in the speech band: level must survive
        out = Resampler(24000).feed(sine(1000, 24000))
        self.assertAlmostEqual(rms(out), rms(sine(1000, 24000)), delta=0.02)

    def test_above_nyquist_tone_suppressed(self):
        # 11 kHz cannot be represented at 16 kHz; without the low-pass it
        # would alias into the speech band instead of disappearing
        out = Resampler(24000).feed(sine(11000, 24000))
        self.assertLess(rms(out), 0.05 * rms(sine(11000, 24000)))

    def test_streaming_equals_batch(self):
        x = sine(700, 24000) + 0.3 * sine(2500, 24000)
        batch = Resampler(24000).feed(x)
        streamed = Resampler(24000)
        pieces, i = [], 0
        for size in (1, 7, 100, 1024, 6000, 6000, 6000, 60):
            pieces.append(streamed.feed(x[i:i + size]))
            i += size
        pieces.append(streamed.feed(x[i:]))
        y = np.concatenate(pieces)
        n = min(len(batch), len(y))
        self.assertGreater(n, TARGET_RATE - Resampler.TAPS)
        np.testing.assert_allclose(y[:n], batch[:n], atol=1e-5)

    def test_16k_passthrough(self):
        x = sine(1000, 16000)
        self.assertIs(Resampler(16000).feed(x), x)


class EchoGateTest(unittest.TestCase):
    """The gate replaces hardware echo cancellation (Apple's voice-processing
    unit silences the mic for every other app — the user goes inaudible in
    their own meeting), so its judgments matter in both directions: bleed of
    the speakers into an open mic must be dropped, the user's own voice must
    never be."""

    def gate_with_reference(self, ref, end_time=1000.0):
        gate = EchoGate()
        # feed as the ~0.25 s chunks the tap delivers, stamped contiguously
        step = TARGET_RATE // 4
        for i in range(0, len(ref), step):
            chunk = ref[i:i + step]
            gate.feed_reference(
                chunk, now=end_time - (len(ref) - i - len(chunk)) / TARGET_RATE)
        return gate

    def setUp(self):
        rng = np.random.default_rng(7)
        self.remote = (0.1 * rng.standard_normal(10 * TARGET_RATE)).astype(np.float32)
        self.rng = rng

    def test_speaker_bleed_is_dropped(self):
        gate = self.gate_with_reference(self.remote, end_time=1000.0)
        # mic hears the speakers 120 ms late, room-filtered and attenuated,
        # over its own noise floor; segment covers wall-clock [995, 998]
        delay = int(0.12 * TARGET_RATE)
        i0 = int((995.0 - 990.0) * TARGET_RATE) - delay
        x = self.remote[i0:i0 + 3 * TARGET_RATE]
        bleed = (0.3 * x + 0.15 * np.roll(x, 80) + 0.05 * np.roll(x, 200)
                 + 0.005 * self.rng.standard_normal(len(x))).astype(np.float32)
        self.assertTrue(gate.is_echo(bleed, now=998.0))

    def test_users_own_voice_is_kept(self):
        gate = self.gate_with_reference(self.remote, end_time=1000.0)
        voice = (0.1 * self.rng.standard_normal(3 * TARGET_RATE)).astype(np.float32)
        self.assertFalse(gate.is_echo(voice, now=998.0))

    def test_voice_over_quiet_speakers_is_kept(self):
        # user talks while faint remote audio bleeds in underneath: the
        # direct voice dominates the segment, so it must survive
        gate = self.gate_with_reference(self.remote, end_time=1000.0)
        i0 = int(5.0 * TARGET_RATE)
        bleed = 0.05 * self.remote[i0:i0 + 3 * TARGET_RATE]
        voice = 0.15 * self.rng.standard_normal(len(bleed))
        self.assertFalse(gate.is_echo((voice + bleed).astype(np.float32), now=998.0))

    def test_silent_speakers_never_trigger(self):
        gate = self.gate_with_reference(np.zeros(10 * TARGET_RATE, dtype=np.float32))
        voice = (0.1 * self.rng.standard_normal(2 * TARGET_RATE)).astype(np.float32)
        self.assertFalse(gate.is_echo(voice, now=998.0))

    def test_no_reference_never_triggers(self):
        # system tap down (permission missing): mic lane must keep flowing
        voice = (0.1 * self.rng.standard_normal(2 * TARGET_RATE)).astype(np.float32)
        self.assertFalse(EchoGate().is_echo(voice, now=998.0))

    def test_segmenter_tees_the_raw_stream(self):
        seen = []
        seg = Segmenter("system", lambda s, a: None, lambda s, r: None,
                        tee=seen.append)
        chunk = np.zeros(1000, dtype=np.float32)
        seg.feed(chunk)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], chunk)


if __name__ == "__main__":
    unittest.main()
