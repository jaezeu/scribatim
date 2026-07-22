"""Tests for the active-speaker heuristics: speaker-view name picking,
tile-strip (screen share / gallery) detection, glow-based attribution, and
the attendee roster. TEAMS_SHARE_FRAME reproduces the geometry of a real
captured Teams meeting with a shared screen (speaker_debug.txt) — same
coordinates, spacing, and OCR artifacts, with every name and product string
replaced by fictional ones."""

import unittest

from scribatim.speaker import (
    SpeakerTracker, _clean, analyze, find_strips, pick_from_strips, pick_name,
)


def ocr(text, x=0.05, y=0.05, conf=0.9, gl=None):
    """One OCR item; defaults put it in the bottom-left name-label region."""
    item = {"text": text, "x": x, "y": y, "w": 0.08, "h": 0.02, "conf": conf}
    if gl:
        item["gl"] = gl
    return item


# Condensed from a real frame: Teams meeting, screen share filling the
# window, participant tiles in a strip across the top (names at y=0.83),
# browser tabs and CRM content from the shared screen below.
TEAMS_SHARE_FRAME = [
    ocr("39:16", x=0.03, y=0.96, conf=1.0),
    ocr("Take contro", x=0.51, y=0.96, conf=1.0),
    ocr("Annotate", x=0.55, y=0.96, conf=1.0),
    ocr("Leave", x=0.96, y=0.96, conf=1.0),
    ocr("Pop out", x=0.58, y=0.95, conf=1.0),
    ocr("People", x=0.66, y=0.95, conf=1.0),
    # the participant tile strip — evenly spaced names
    ocr("Mei Ling Tan", x=0.03, y=0.83, conf=1.0),
    ocr("Amanda Wei Lin Goh*", x=0.13, y=0.83, conf=1.0),
    ocr("Zhenwei (Kevin) L..", x=0.23, y=0.83, conf=1.0),
    ocr("Rohan Deshpande", x=0.33, y=0.83, conf=1.0),
    ocr("Nikhil Sathe", x=0.44, y=0.83, conf=1.0),
    ocr("Suresh Pawar", x=0.54, y=0.83, conf=1.0),
    ocr("Kevin Vu", x=0.64, y=0.83, conf=1.0),
    ocr("Ren Takahashi", x=0.75, y=0.83, conf=1.0),
    # shared screen: browser tabs (name-ish but irregularly spaced)
    ocr("My Apps Dashboard X", x=0.08, y=0.78, conf=1.0),
    ocr("ACMEX Scrum Boar", x=0.28, y=0.78, conf=1.0),
    ocr("Acme - OneDrive", x=0.54, y=0.78, conf=1.0),
    # shared screen: CRM table content, incl. a person name bottom-left
    ocr("External Resources", x=0.12, y=0.74, conf=1.0),
    ocr("Mark Reynolds", x=0.29, y=0.28, conf=1.0),
    ocr("•COlL", x=0.10, y=0.81, conf=0.3),
]

ROSTER = ["Mei Ling Tan", "Amanda Wei Lin Goh", "Zhenwei L",
          "Rohan Deshpande", "Nikhil Sathe", "Suresh Pawar",
          "Kevin Vu", "Ren Takahashi"]


def with_glow(frame, speaking, gl=(0.3, 0.28, 258, 252)):
    """The frame with an active-speaker outline around one tile."""
    out = []
    for item in frame:
        item = dict(item)
        if item["text"] == speaking:
            item["gl"] = list(gl)
        out.append(item)
    return out


class Clean(unittest.TestCase):
    def test_teams_mic_icon_artifacts_stripped(self):
        # OCR renders the mic icon next to Teams names as stray symbols
        self.assertEqual(_clean("Amanda Wei Lin Goh*"), "Amanda Wei Lin Goh")
        self.assertEqual(_clean("Amanda Wei Lin Goh &"), "Amanda Wei Lin Goh")

    def test_truncation_ellipsis_stripped(self):
        self.assertEqual(_clean("Zhenwei (Kevin) L.."), "Zhenwei L")
        self.assertEqual(_clean("Zhenwei (Kevin) L..."), "Zhenwei L")

    def test_parenthetical_removed(self):
        self.assertEqual(_clean("María López (Host, me)"), "María López")


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


class FindStrips(unittest.TestCase):
    def test_teams_share_frame_yields_the_participant_strip(self):
        strips = find_strips(TEAMS_SHARE_FRAME)
        self.assertEqual(len(strips), 1)
        self.assertEqual([n for n, _ in strips[0]], ROSTER)

    def test_browser_tab_row_is_not_a_strip(self):
        # only the shared-screen rows, no participant strip
        junk = [t for t in TEAMS_SHARE_FRAME if t["y"] != 0.83]
        self.assertEqual(find_strips(junk), [])

    def test_filter_chip_row_is_not_a_strip(self):
        # regression from a real capture: a shared CRM screen's filter chips
        # are evenly spaced and name-ish, but not Title Case display names
        chips = ["Filters", "x Deal Status", "Revenue", "x Test Accounty",
                 "x Exclude OEM", "x User Theater"]
        row = [ocr(t, x=0.16 + i * 0.055, y=0.66, conf=1.0)
               for i, t in enumerate(chips)]
        self.assertEqual(find_strips(row), [])

    def test_crm_table_row_with_names_is_not_a_strip(self):
        # regression from a real capture: one table row can be evenly spaced
        # Title Case names — but table rows stack densely, tiles don't
        table = [
            ocr("Kenta Mori", x=0.29, y=0.55, conf=1.0),
            ocr("Field Sales - Japan", x=0.38, y=0.55, conf=1.0),
            ocr("USU", x=0.62, y=0.55, conf=1.0),
            ocr("USU", x=0.72, y=0.55, conf=1.0),
            ocr("FroDable", x=0.83, y=0.55, conf=1.0),
            # the neighboring table row, 0.03 below
            ocr("Mark Reynolds", x=0.29, y=0.58, conf=1.0),
            ocr("USD", x=0.62, y=0.58, conf=1.0),
            ocr("REnewal Due", x=0.88, y=0.58, conf=1.0),
        ]
        self.assertEqual(find_strips(table), [])

    def test_three_names_are_not_enough_evidence(self):
        row = [ocr(n, x=0.1 + i * 0.2, y=0.85) for i, n in enumerate(
            ["Alice Wong", "Bob Ray", "Carol Diaz"])]
        self.assertEqual(find_strips(row), [])

    def test_one_ocr_missed_label_tolerated(self):
        # a gap of ~2× the median (one unread tile) must not break the strip
        xs = [0.03, 0.13, 0.33, 0.43, 0.53]  # missing label at 0.23
        row = [ocr(f"Name Number{i}", x=x, y=0.85) for i, x in enumerate(xs)]
        self.assertEqual(len(find_strips(row)), 1)


class PickFromStrips(unittest.TestCase):
    def strips(self, frame):
        return find_strips(frame)

    def test_no_glow_abstains(self):
        self.assertIsNone(pick_from_strips(self.strips(TEAMS_SHARE_FRAME)))

    def test_glowing_tile_wins(self):
        frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")
        self.assertEqual(pick_from_strips(self.strips(frame)), "Suresh Pawar")

    def test_weak_glow_abstains(self):
        frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar", gl=(0.05, 0.04, 258, 252))
        self.assertIsNone(pick_from_strips(self.strips(frame)))

    def test_two_similar_glows_abstain(self):
        frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")
        frame = with_glow(frame, "Kevin Vu", gl=(0.25, 0.22, 190, 200))
        self.assertIsNone(pick_from_strips(self.strips(frame)))

    def test_mismatched_band_hues_rejected(self):
        # left band purple, bottom band green = video content, not an outline
        frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar", gl=(0.3, 0.28, 258, 110))
        self.assertIsNone(pick_from_strips(self.strips(frame)))


class Analyze(unittest.TestCase):
    def test_share_frame_gives_roster_without_guessing_speaker(self):
        speaker, roster = analyze(TEAMS_SHARE_FRAME)
        self.assertIsNone(speaker)
        self.assertEqual(roster, ROSTER)

    def test_shared_content_name_never_becomes_the_speaker(self):
        # "Mark Reynolds" sits bottom-left but belongs to the shared CRM table
        speaker, _ = analyze(TEAMS_SHARE_FRAME)
        self.assertNotEqual(speaker, "Mark Reynolds")

    def test_glow_names_the_speaker(self):
        speaker, roster = analyze(with_glow(TEAMS_SHARE_FRAME, "Ren Takahashi"))
        self.assertEqual(speaker, "Ren Takahashi")
        self.assertEqual(roster, ROSTER)

    def test_speaker_view_still_uses_bottom_left_label(self):
        speaker, roster = analyze([ocr("Kenji Tanaka")])
        self.assertEqual(speaker, "Kenji Tanaka")
        self.assertEqual(roster, [])


class Roster(unittest.TestCase):
    def test_persistent_strip_names_enter_the_roster(self):
        tracker = SpeakerTracker()
        for i in range(10):
            tracker._ingest({"time": 1000.0 + i, "texts": TEAMS_SHARE_FRAME})
        self.assertEqual(sorted(tracker.roster()), sorted(ROSTER))

    def test_one_frame_misread_stays_out(self):
        tracker = SpeakerTracker()
        for i in range(10):
            tracker._ingest({"time": 1000.0 + i, "texts": TEAMS_SHARE_FRAME})
        glitch = TEAMS_SHARE_FRAME + [ocr("Garbled Namex", x=0.85, y=0.83)]
        tracker._ingest({"time": 1010.0, "texts": glitch})
        self.assertNotIn("Garbled Namex", tracker.roster())

    def test_speaker_view_names_accumulate_too(self):
        tracker = SpeakerTracker()
        for i in range(5):
            tracker._ingest({"time": 1000.0 + i, "texts": [ocr("Kenji Tanaka")]})
        self.assertEqual(tracker.roster(), ["Kenji Tanaka"])

    def test_votes_feed_name_for(self):
        tracker = SpeakerTracker()
        for i in range(6):
            frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")
            tracker._ingest({"time": 1000.0 + i, "texts": frame})
        self.assertEqual(tracker.name_for(1000.0, 1006.0), "Suresh Pawar")

    def test_name_carried_over_a_glow_flicker(self):
        # the outline faded between sentences: no samples inside the caption's
        # window, but the same person still has the floor
        tracker = SpeakerTracker()
        for i in range(6):
            frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")
            tracker._ingest({"time": 1000.0 + i, "texts": frame})
        self.assertEqual(tracker.name_for(1008.0, 1012.0), "Suresh Pawar")

    def test_name_not_carried_over_a_long_gap(self):
        tracker = SpeakerTracker()
        for i in range(6):
            frame = with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")
            tracker._ingest({"time": 1000.0 + i, "texts": frame})
        self.assertIsNone(tracker.name_for(1030.0, 1034.0))

    def test_window_votes_beat_carried_name(self):
        tracker = SpeakerTracker()
        tracker._ingest({"time": 1000.0,
                         "texts": with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")})
        for i in range(3):
            tracker._ingest({"time": 1004.0 + i,
                             "texts": with_glow(TEAMS_SHARE_FRAME, "Kevin Vu")})
        self.assertEqual(tracker.name_for(1003.0, 1008.0), "Kevin Vu")

    def test_lingering_previous_glow_does_not_steal_the_turn(self):
        # Suresh stops talking; his outline keeps fading into the first
        # second of Kevin's reply. The window-start samples still show
        # Suresh — they must not outvote Kevin's samples later in the turn.
        tracker = SpeakerTracker()
        for t in (1005.0, 1006.0, 1007.0):
            tracker._ingest({"time": t,
                             "texts": with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")})
        for t in (1008.0, 1009.0):
            tracker._ingest({"time": t,
                             "texts": with_glow(TEAMS_SHARE_FRAME, "Kevin Vu")})
        self.assertEqual(tracker.name_for(1005.7, 1009.0), "Kevin Vu")

    def test_glow_arriving_after_the_words_names_the_speaker(self):
        # short utterance: the app draws Kevin's outline only after the
        # caption window closed. The trailing sample must beat carrying the
        # previous speaker's name backward over the turn change.
        tracker = SpeakerTracker()
        tracker._ingest({"time": 998.0,
                         "texts": with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")})
        tracker._ingest({"time": 1004.2,
                         "texts": with_glow(TEAMS_SHARE_FRAME, "Kevin Vu")})
        self.assertEqual(tracker.name_for(1000.0, 1003.0), "Kevin Vu")

    def test_next_speakers_reply_does_not_steal_the_caption(self):
        # Suresh's own turn has in-window samples; Kevin's outline lighting
        # up right after the window (his reply) carries less weight.
        tracker = SpeakerTracker()
        for t in (1002.0, 1003.0):
            tracker._ingest({"time": t,
                             "texts": with_glow(TEAMS_SHARE_FRAME, "Suresh Pawar")})
        tracker._ingest({"time": 1005.5,
                         "texts": with_glow(TEAMS_SHARE_FRAME, "Kevin Vu")})
        self.assertEqual(tracker.name_for(1000.0, 1004.0), "Suresh Pawar")


if __name__ == "__main__":
    unittest.main()
