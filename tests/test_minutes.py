"""Smoke tests for the transcript formatting and map-reduce chunking in
susurro.minutes — the pure-logic pieces that would regress silently."""

import unittest
from unittest import mock

from susurro import minutes


def caption(text="hello", source="system", speaker=None, lang="en", t=1751844000.0):
    return {"time": t, "source": source, "speaker": speaker,
            "lang": lang, "text_en": text}


class TranscriptToText(unittest.TestCase):
    def test_mic_becomes_you(self):
        line = minutes.transcript_to_text([caption(source="mic")])
        self.assertIn("] You: hello", line)

    def test_speaker_name_used_when_known(self):
        line = minutes.transcript_to_text([caption(speaker="Kenji Tanaka")])
        self.assertIn("] Kenji Tanaka: hello", line)

    def test_unknown_speaker_is_participants(self):
        line = minutes.transcript_to_text([caption()])
        self.assertIn("] Participants: hello", line)

    def test_non_english_gets_lang_tag(self):
        line = minutes.transcript_to_text([caption(lang="ja")])
        self.assertIn("[ja→en]", line)
        self.assertNotIn("[en→en]", minutes.transcript_to_text([caption()]))

    def test_one_line_per_caption_in_order(self):
        text = minutes.transcript_to_text(
            [caption(text=f"line {i}") for i in range(5)])
        lines = text.split("\n")
        self.assertEqual(len(lines), 5)
        self.assertIn("line 0", lines[0])
        self.assertIn("line 4", lines[4])


class SplitLines(unittest.TestCase):
    def test_single_part_when_under_budget(self):
        lines = ["short line"] * 3
        self.assertEqual(len(minutes._split_lines(lines, budget=1000)), 1)

    def test_splits_and_preserves_every_line(self):
        lines = [f"line number {i} with some padding text" for i in range(100)]
        parts = minutes._split_lines(lines, budget=50)
        self.assertGreater(len(parts), 1)
        self.assertEqual("\n".join(parts).split("\n"), lines)  # nothing lost

    def test_each_part_fits_budget(self):
        lines = ["x" * 30 for _ in range(50)]
        for part in minutes._split_lines(lines, budget=40):
            self.assertLessEqual(
                sum(minutes._estimate_tokens(l) + 1 for l in part.split("\n")), 40)

    def test_oversized_single_line_still_emitted(self):
        parts = minutes._split_lines(["y" * 500], budget=10)
        self.assertEqual(parts, ["y" * 500])


class Generate(unittest.TestCase):
    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            minutes.generate({}, "poem", [caption()])

    def test_empty_transcript_rejected(self):
        with self.assertRaises(RuntimeError):
            minutes.generate({}, "minutes", [])

    def test_short_meeting_single_llm_call(self):
        cfg = {"ollama_model": "test", "llm_num_ctx": 8192}
        with mock.patch.object(minutes, "_chat", return_value="## Done") as chat:
            out = minutes.generate(cfg, "minutes", [caption()])
        self.assertEqual(out, "## Done")
        self.assertEqual(chat.call_count, 1)
        self.assertIn("hello", chat.call_args[0][1])  # transcript reached the prompt

    def test_long_meeting_is_condensed_not_truncated(self):
        cfg = {"ollama_model": "test", "llm_num_ctx": 4096}
        caps = [caption(text=f"utterance {i}: " + "words " * 40, t=1751844000 + i)
                for i in range(400)]
        with mock.patch.object(minutes, "_chat", return_value="### notes") as chat:
            minutes.generate(cfg, "summary", caps)
        # chunk passes plus the final deliverable pass
        self.assertGreater(chat.call_count, 1)
        final_prompt = chat.call_args[0][1]
        self.assertIn("Condensed notes", final_prompt)


if __name__ == "__main__":
    unittest.main()
