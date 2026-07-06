"""Smoke tests for the Whisper hallucination filter."""

import unittest

import numpy as np

from susurro.transcriber import HALLUCINATION_RMS, is_hallucination


def tone(rms):
    """1s of 16 kHz noise scaled to a given RMS level."""
    rng = np.random.default_rng(seed=1)
    x = rng.standard_normal(16000).astype(np.float32)
    return x / np.sqrt(np.mean(x ** 2)) * rms


QUIET = tone(HALLUCINATION_RMS / 4)
LOUD = tone(HALLUCINATION_RMS * 5)


class IsHallucination(unittest.TestCase):
    def test_stock_phrase_on_quiet_audio_dropped(self):
        self.assertTrue(is_hallucination("Thank you.", QUIET))

    def test_punctuation_and_case_variants_dropped(self):
        for text in ("thank you", "Thank you!", " Thanks for watching. ", "Bye!"):
            self.assertTrue(is_hallucination(text, QUIET), text)

    def test_same_phrase_spoken_loudly_kept(self):
        # a real "thank you" said directly into the mic must survive
        self.assertFalse(is_hallucination("Thank you.", LOUD))

    def test_real_sentence_on_quiet_audio_kept(self):
        self.assertFalse(is_hallucination(
            "Thank you, and let's move the deadline to Friday.", QUIET))

    def test_phrase_embedded_in_sentence_kept(self):
        self.assertFalse(is_hallucination("I want to subscribe to the API plan.", QUIET))


if __name__ == "__main__":
    unittest.main()
