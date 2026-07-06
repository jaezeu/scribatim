"""Post-meeting deliverables via a local Ollama model. Localhost only.

Three generators share one transcript format:
  minutes  — full structured minutes with action items
  summary  — quick "where are we" recap, usable mid-meeting
  email    — ready-to-send follow-up email draft
"""

import logging
import time

import requests

log = logging.getLogger("susurro.minutes")

_CONTEXT = """You are an expert meeting assistant.{context} \
Below is a meeting transcript. Lines are labeled "You" (the person running \
this tool) and "Participants" (everyone else on the call); non-English speech \
has been translated to English. \
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
A Markdown table: | # | Action | Owner | Due |. Infer owners ("You" or \
"Participants") and any dates mentioned; use "TBD" when not stated.
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


def transcript_to_text(captions: list[dict]) -> str:
    lines = []
    for c in captions:
        stamp = time.strftime("%H:%M:%S", time.localtime(c["time"]))
        who = "You" if c["source"] == "mic" else "Participants"
        lang = f" [{c['lang']}→en]" if c["lang"] != "en" else ""
        lines.append(f"[{stamp}] {who}{lang}: {c['text_en']}")
    return "\n".join(lines)


def generate(cfg: dict, kind: str, captions: list[dict]) -> str:
    if kind not in PROMPTS:
        raise ValueError(f"unknown deliverable: {kind}")
    if not captions:
        raise RuntimeError("no transcript captured yet")
    ctx = (cfg.get("meeting_context") or "").strip()
    prompt = PROMPTS[kind].format(
        context=f" {ctx}" if ctx else "",
        date=time.strftime("%Y-%m-%d %H:%M"),
        transcript=transcript_to_text(captions))
    log.info("generating %s with %s (%d captions)…", kind, cfg["ollama_model"], len(captions))
    try:
        resp = requests.post(
            f"{cfg['ollama_url']}/api/chat",
            json={
                "model": cfg["ollama_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 8192},
            },
            timeout=600)
        resp.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError(
            "Ollama is not reachable on 127.0.0.1:11434 — start it with "
            "`brew services start ollama`") from e
    return resp.json()["message"]["content"].strip()
