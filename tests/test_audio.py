"""Tests for the streaming anti-aliased resampler."""

import unittest

import numpy as np

from scribatim.audio import TARGET_RATE, Resampler


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


if __name__ == "__main__":
    unittest.main()
