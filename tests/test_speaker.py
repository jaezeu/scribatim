"""Smoke tests for the active-speaker name-picking heuristics."""

import unittest

from susurro.speaker import pick_name


def ocr(text, x=0.05, y=0.05, conf=0.9):
    """One OCR item; defaults put it in the bottom-left name-label region."""
    return {"text": text, "x": x, "y": y, "conf": conf}


class PickName(unittest.TestCase):
    def test_picks_bottom_left_label(self):
        self.assertEqual(pick_name([ocr("Kenji Tanaka")]), "Kenji Tanaka")

    def test_ui_noise_rejected(self):
        for noise in ("Mute", "Share Screen", "Leave", "Gallery View", "Zoom"):
            self.assertIsNone(pick_name([ocr(noise)]), noise)

    def test_clocks_and_counters_rejected(self):
        for junk in ("10:23", "00:41:07", "12 participants", "http://zoom.us"):
            self.assertIsNone(pick_name([ocr(junk)]), junk)

    def test_labels_outside_bottom_left_ignored(self):
        self.assertIsNone(pick_name([ocr("Kenji Tanaka", y=0.9)]))   # top bar
        self.assertIsNone(pick_name([ocr("Kenji Tanaka", x=0.8)]))   # right side

    def test_low_confidence_ignored(self):
        self.assertIsNone(pick_name([ocr("Kenji Tanaka", conf=0.1)]))

    def test_host_suffix_stripped(self):
        self.assertEqual(pick_name([ocr("María López (Host, me)")]), "María López")

    def test_closest_to_corner_wins(self):
        got = pick_name([ocr("Far Name", x=0.5, y=0.3), ocr("Near Name", x=0.02, y=0.03)])
        self.assertEqual(got, "Near Name")

    def test_gallery_view_abstains(self):
        # >3 plausible name labels = gallery view; guessing would be wrong
        frame = [ocr(n, x=0.05 + i * 0.05) for i, n in enumerate(
            ["Alice Wong", "Bob Ray", "Carol Diaz", "Dan Oz"])]
        self.assertIsNone(pick_name(frame))

    def test_cjk_names_accepted(self):
        self.assertEqual(pick_name([ocr("田中健二")]), "田中健二")

    def test_empty_frame(self):
        self.assertIsNone(pick_name([]))


if __name__ == "__main__":
    unittest.main()
