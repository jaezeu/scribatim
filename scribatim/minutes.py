"""Post-meeting deliverables via a local Ollama model. Localhost only.

Three generators share one transcript format:
  minutes  — full structured minutes with action items
  summary  — quick "where are we" recap, usable mid-meeting
  email    — ready-to-send follow-up email draft

Long meetings don't fit the model's context window, and Ollama silently drops
the *start* of an oversized prompt — exactly where the agenda and early
decisions live. So generation is map-reduce: transcripts that fit go through
in one pass; longer ones are compressed chunk-by-chunk into dense notes
(keeping names, dates, numbers, commitments), and the deliverable is written
from the notes.
"""

import logging
import time

import requests

log = logging.getLogger("scribatim.minutes")

_CONTEXT = """You are an expert meeting assistant.{context} \
Below is a meeting transcript. Lines are labeled "You" (the person running \
this tool); other speakers appear by name when known, otherwise as \
"Participants".{attendees} Non-English speech has been translated to English. \
Be faithful to the transcript — never invent facts, names, or dates. \
Transcription may contain small errors; smooth over obvious ones.
"""

PROMPTS = {
    "minutes": _CONTEXT + """
Write crisp meeting minutes in Markdown with exactly these sections:

# Meeting Minutes — {date}
## Summary
3-6 sentences: who met, why, what was covered, overall outcome.
## Key Discussion Points
Bullet list of the substantive topics (products, use cases, objections, requirements).
## Decisions
Bullet list of anything agreed or concluded. Write "None recorded" if empty.
## Action Items
A Markdown table: | # | Action | Owner | Due |. Infer owners (speaker names \
when present, else "You"/"Participants") and dates mentioned; "TBD" when unstated.
## Open Questions / Risks
Bullets for unresolved questions, blockers, or risks worth following up.

TRANSCRIPT:
{transcript}
""",
    "summary": _CONTEXT + """
Write a quick mid-meeting recap in Markdown, short enough to scan in 20 seconds:

## Recap so far — {date}
- 4-8 bullets covering what has been discussed and any positions taken
- **Bold** anything that sounds like a commitment, date, or decision
## Live threads
- 1-4 bullets: questions currently open or topics still unresolved

TRANSCRIPT:
{transcript}
""",
    "email": _CONTEXT + """
Draft a follow-up email from "You" to the other participants. Markdown, \
with a "Subject:" line first. Tone: warm, concise, professional. \
Structure: one-line thank-you; 2-4 bullets \
recapping what was agreed or clarified; a short "Next steps" list with owners \
and dates from the transcript (TBD if unstated); sign-off placeholder "[Your name]".

TRANSCRIPT:
{transcript}
""",
}


CHUNK_PROMPT = _CONTEXT + """
This is part {part} of {parts} of a long meeting. Compress it into dense notes
for a later summarization pass, under exactly these headings:
### Discussion
### Decisions
### Action items (with owner and due date)
### Open questions
Keep every name, number, date, and commitment. Drop greetings and filler.
Write "None" under a heading with nothing to report.

TRANSCRIPT PART {part}/{parts}:
{transcript}
"""


def transcript_to_text(captions: list[dict]) -> str:
    lines = []
    for c in captions:
        stamp = time.strftime("%H:%M:%S", time.localtime(c["time"]))
        who = "You" if c["source"] == "mic" else (c.get("speaker") or "Participants")
        lang = f" [{c['lang']}→en]" if c["lang"] != "en" else ""
        lines.append(f"[{stamp}] {who}{lang}: {c['text_en']}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token for English prose; timestamps and names tokenize worse,
    # so use 3 to err toward chunking rather than silent truncation
    return len(text) // 3


def _input_budget(cfg: dict) -> int:
    # leave headroom in the context window for the prompt and the response
    return max(1024, int(cfg.get("llm_num_ctx", 8192)) - 2800)


def _split_lines(lines: list[str], budget: int) -> list[str]:
    """Pack lines into the fewest parts that each fit the token budget."""
    parts, current, size = [], [], 0
    for line in lines:
        t = _estimate_tokens(line) + 1
        if current and size + t > budget:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += t
    if current:
        parts.append("\n".join(current))
    return parts


def _chat(cfg: dict, prompt: str) -> str:
    try:
        resp = requests.post(
            f"{cfg['ollama_url']}/api/chat",
            json={
                "model": cfg["ollama_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2,
                            "num_ctx": int(cfg.get("llm_num_ctx", 8192))},
            },
            timeout=600)
        resp.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError(
            "Ollama is not reachable on 127.0.0.1:11434 — start it with "
            "`brew services start ollama`") from e
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", "")
        except Exception:
            pass
        raise RuntimeError(f"Ollama error: {detail or e}") from e
    return resp.json()["message"]["content"].strip()


def _condense(cfg: dict, context: str, attendees: str, text: str) -> str:
    """Compress an oversized transcript into notes that fit the budget."""
    budget = _input_budget(cfg)
    for round_no in range(1, 4):  # 3 rounds ≈ days of audio; loop guard
        parts = _split_lines(text.split("\n"), budget)
        if len(parts) == 1:
            break
        log.info("transcript over context budget — condensing %d parts (round %d)",
                 len(parts), round_no)
        notes = [
            CHUNK_PROMPT.format(context=context, attendees=attendees,
                                part=i, parts=len(parts), transcript=part)
            for i, part in enumerate(parts, 1)]
        text = "\n\n".join(
            f"--- Notes from part {i}/{len(parts)} ---\n{_chat(cfg, note)}"
            for i, note in enumerate(notes, 1))
    if _estimate_tokens(text) > budget:
        log.warning("notes still exceed the context budget after condensing — "
                    "the start of the meeting may be under-represented")
    return text


def generate(cfg: dict, kind: str, captions: list[dict],
             attendees: list[str] | None = None) -> str:
    if kind not in PROMPTS:
        raise ValueError(f"unknown deliverable: {kind}")
    if not captions:
        raise RuntimeError("no transcript captured yet")
    ctx = (cfg.get("meeting_context") or "").strip()
    context = f" {ctx}" if ctx else ""
    # names read off the meeting window (speaker OCR): even when individual
    # captions aren't attributed, the model can name owners and attendees
    attendees_line = (
        f' Attendees seen in the meeting window: {", ".join(attendees)}. '
        'Attribute statements to them only when the transcript makes it clear '
        'who is speaking.') if attendees else ""
    log.info("generating %s with %s (%d captions)…", kind, cfg["ollama_model"], len(captions))

    transcript = transcript_to_text(captions)
    if _estimate_tokens(transcript) > _input_budget(cfg):
        condensed = _condense(cfg, context, attendees_line, transcript)
        transcript = ("Condensed notes from a long meeting "
                      "(compiled chronologically):\n\n" + condensed)

    prompt = PROMPTS[kind].format(
        context=context,
        attendees=attendees_line,
        date=time.strftime("%Y-%m-%d %H:%M"),
        transcript=transcript)
    return _chat(cfg, prompt)
