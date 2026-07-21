"""Smoke tests for the Whisper hallucination filter."""

import unittest

import numpy as np

from scribatim.transcriber import (
    HALLUCINATION_RMS, _suspect_segment, is_hallucination, normalize_segment,
)


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

    def test_word_loop_on_quiet_audio_dropped(self):
        # the classic stuck-decoder loop on a muted/silent mic
        self.assertTrue(is_hallucination("demokrat! demokrat! demokrat!", QUIET))

    def test_extreme_word_loop_dropped_even_when_loud(self):
        self.assertTrue(is_hallucination(
            "demokrat demokrat demokrat demokrat demokrat demokrat", LOUD))

    def test_long_repeated_phrase_dropped(self):
        self.assertTrue(is_hallucination(
            "please like and subscribe " * 6, LOUD))

    def test_short_real_repetition_kept_when_loud(self):
        # a human "no, no, no" said directly into the mic must survive
        self.assertFalse(is_hallucination("No, no, no.", LOUD))

    def test_normal_sentence_kept_when_loud(self):
        self.assertFalse(is_hallucination(
            "The migration is scheduled for the second week of August.", LOUD))


class SuspectSegment(unittest.TestCase):
    def test_confident_speech_kept(self):
        self.assertFalse(_suspect_segment(-0.3, 0.1, 1.4))

    def test_text_decoded_over_silence_dropped(self):
        self.assertTrue(_suspect_segment(-1.5, 0.9, 1.4))

    def test_silencelike_but_confident_kept(self):
        # high no-speech prob alone is not enough — Whisper's own rule
        self.assertFalse(_suspect_segment(-0.3, 0.9, 1.4))

    def test_repetition_loop_dropped(self):
        self.assertTrue(_suspect_segment(-0.3, 0.1, 3.1))

    def test_strict_drops_on_any_weak_signal(self):
        # segments the VAD distrusts: one mediocre signal is enough
        self.assertTrue(_suspect_segment(-0.9, 0.1, 1.4, strict=True))
        self.assertTrue(_suspect_segment(-0.3, 0.55, 1.4, strict=True))
        self.assertTrue(_suspect_segment(-0.3, 0.1, 2.3, strict=True))

    def test_strict_keeps_confident_speech(self):
        self.assertFalse(_suspect_segment(-0.3, 0.1, 1.4, strict=True))


class NormalizeSegment(unittest.TestCase):
    def test_quiet_speech_boosted(self):
        x = tone(0.01)
        y = normalize_segment(x)
        self.assertGreater(np.max(np.abs(y)), np.max(np.abs(x)) * 2)

    def test_gain_is_capped(self):
        x = tone(0.0005)
        y = normalize_segment(x)
        self.assertLessEqual(np.max(np.abs(y)), np.max(np.abs(x)) * 12 + 1e-6)

    def test_near_silence_untouched(self):
        x = np.full(16000, 1e-5, dtype=np.float32)
        self.assertIs(normalize_segment(x), x)

    def test_loud_audio_untouched(self):
        x = tone(0.3)
        self.assertIs(normalize_segment(x), x)


if __name__ == "__main__":
    unittest.main()
