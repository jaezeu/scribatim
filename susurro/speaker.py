"""Active-speaker names via screen OCR (experimental, off by default).

Reads JSON lines from bin/susurro-speaker (ScreenCaptureKit + Vision — fully
on-device), decides which recognized text is the active speaker's name label,
and answers "who was speaking during [t0, t1]?" by majority vote over the
samples in that window.

Heuristics target Zoom/Teams *speaker view*, where the active speaker fills
the window and their name label sits near the bottom-left. Gallery view shows
many equally plausible labels, so we return nothing rather than guess wrong.
Vision coordinates are normalized with origin at the bottom-left.
"""

import json
import logging
import re
import subprocess
import threading
import time
from collections import deque

from .audio import BIN_DIR

log = logging.getLogger("susurro.speaker")

SPEAKER_BINARY = BIN_DIR / "susurro-speaker"

# Meeting-app UI strings that OCR picks up but are never names.
UI_NOISE = {
    # shared / Zoom
    "mute", "unmute", "muted", "stop video", "start video", "share", "share screen",
    "stop share", "record", "recording", "reactions", "view", "chat", "participants",
    "people", "more", "leave", "end", "end meeting", "meeting", "join", "invite",
    "security", "apps", "notes", "rooms", "whiteboards", "captions", "copilot",
    "camera", "mic", "audio", "video", "raise", "raise hand", "lower hand",
    "speaker view", "gallery view", "zoom", "zoom meeting", "zoom workplace",
    "microsoft teams", "teams", "host", "co-host", "me", "you", "guest",
    "waiting room", "pause", "resume", "settings",
    # Teams desktop
    "react", "present", "meet", "meet now", "breakout rooms", "together mode",
    "focus", "attendance", "hold", "device settings", "show conversation",
    "hide conversation", "turn camera on", "turn camera off", "leave meeting",
    "take control", "give control", "waiting for others to join", "in a call",
    "presenter", "attendee", "organizer", "spotlight", "pin", "fit to frame",
}

# Starts with a letter (any script, incl. CJK), then letters/marks/dots/
# hyphens/apostrophes/spaces. Rejects clocks, counters, URLs.
NAMEISH = re.compile(r"^[^\W\d_][\w.'’\- ]{0,39}$")


def _clean(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)  # "(Host, me)" → ""
    return text.strip(" .,:;|•·-–—")


def pick_name(texts: list[dict]) -> str | None:
    """Choose the active speaker's name label from one frame's OCR results."""
    candidates = []
    for item in texts:
        name = _clean(item.get("text", ""))
        if not name or len(name) > 40 or len(name.split()) > 4:
            continue
        if name.lower() in UI_NOISE or not NAMEISH.match(name):
            continue
        if item.get("conf", 1.0) < 0.3:
            continue
        x, y = item.get("x", 1.0), item.get("y", 1.0)
        # name labels live in the bottom-left region of the speaker tile
        if y > 0.35 or x > 0.6:
            continue
        candidates.append((x * x + y * y, name))  # distance to bottom-left corner
    # >3 plausible labels means gallery view / ambiguity — abstain, don't guess
    if not candidates or len(candidates) > 3:
        return None
    return min(candidates)[1]


class SpeakerTracker:
    """Runs the OCR helper and keeps a rolling (time, name) timeline."""

    def __init__(self):
        self.samples: deque = deque(maxlen=1800)  # ~30 min at 1 Hz
        self.proc: subprocess.Popen | None = None

    def start(self):
        if self.proc:
            return
        if not SPEAKER_BINARY.exists():
            raise RuntimeError(f"speaker helper missing: {SPEAKER_BINARY} — run setup.sh")
        self.proc = subprocess.Popen(
            [str(SPEAKER_BINARY)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for target in (self._read_frames, self._read_stderr):
            threading.Thread(target=target, daemon=True).start()
        log.info("speaker OCR running (experimental)")

    def _read_frames(self):
        proc = self.proc
        assert proc and proc.stdout
        for line in proc.stdout:
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            name = pick_name(frame.get("texts", []))
            if name:
                self.samples.append((float(frame.get("time", time.time())), name))

    def _read_stderr(self):
        proc = self.proc
        assert proc and proc.stderr
        for line in proc.stderr:
            log.info("speaker: %s", line.decode(errors="replace").rstrip())

    def name_for(self, t0: float, t1: float) -> str | None:
        """Majority-vote name over samples inside [t0, t1] (±1s slack)."""
        votes: dict[str, int] = {}
        for t, name in list(self.samples):
            if t0 - 1.0 <= t <= t1 + 1.0:
                votes[name] = votes.get(name, 0) + 1
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
