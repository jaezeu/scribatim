"""Active-speaker names via screen OCR (experimental, off by default).

Reads JSON lines from bin/scribatim-speaker (ScreenCaptureKit + Vision — fully
on-device), decides which recognized text is the active speaker's name label,
and answers "who was speaking during [t0, t1]?" by majority vote over the
samples in that window.

Two layouts are recognized:

* Speaker view — the active speaker fills the window and their name label
  sits near the bottom-left. We pick the name closest to that corner.
* Tile strip / gallery — during a screen share (and in gallery view) the
  apps show a row of equally sized participant tiles, each labeled at its
  bottom-left. Every label is visible all the time, so the text alone can't
  say who is talking; what does change is the colored outline the app draws
  around the speaking tile. The Swift helper samples the pixels just left of
  and just below each label ("glow" bands) and we pick the label whose bands
  clearly light up in one consistent hue. The strip also gives us the
  attendee roster, which feeds the minutes generator.

Vision coordinates are normalized with origin at the bottom-left.
"""

import json
import logging
import re
import subprocess
import threading
import time
from collections import Counter, deque

from .audio import BIN_DIR

log = logging.getLogger("scribatim.speaker")

SPEAKER_BINARY = BIN_DIR / "scribatim-speaker"

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
    # screen-share toolbars
    "annotate", "pop out", "paused", "presenting", "stop presenting",
    "you're presenting", "stop sharing",
}

# Starts with a letter (any script, incl. CJK), then letters/marks/dots/
# hyphens/apostrophes/spaces. Rejects clocks, counters, URLs.
NAMEISH = re.compile(r"^[^\W\d_][\w.'’\- ]{0,39}$")

# Tile-strip detection: a row of evenly spaced name labels. Three labels give
# only two gaps — too little evidence of regularity — so require four.
STRIP_MIN_NAMES = 4
STRIP_Y_TOL = 0.015      # labels in one strip share a baseline
STRIP_GAP_TOL = 0.35     # each x-gap within ±35% of the median gap
                         # (or of 2× the median: one label OCR missed)
STRIP_ROW_MIN_DY = 0.045  # another name-ish row closer than this = a text
                          # table (tiles are tall; table rows are dense)

# Glow (active-speaker outline) thresholds. "gl" per label is
# [left_frac, bottom_frac, left_hue, bottom_hue]: the fraction of vividly
# colored pixels in the band left of / below the label, and their mean hue.
GLOW_MIN = 0.10          # the winner's weaker band must be at least this lit
GLOW_DOMINANCE = 2.2     # ...and clearly ahead of the runner-up
GLOW_HUE_TOL = 55.0      # both bands must agree it's one outline color


def _clean(text: str) -> str:
    text = re.sub(r"\(.*?\)", " ", text)      # "(Host, me)" → ""
    text = re.sub(r"[.…]{2,}\s*$", "", text)  # OCR'd truncation: "Kenji W.."
    # trailing "*"/"&" are how OCR renders the mic icon next to Teams names
    text = text.strip(" .,:;|•·-–—*&+")
    return re.sub(r"\s{2,}", " ", text).strip()


def _nameish(item: dict) -> str | None:
    """The cleaned text if this OCR item plausibly is a name label."""
    name = _clean(item.get("text", ""))
    if not name or len(name) > 40 or len(name.split()) > 4:
        return None
    if name.lower() in UI_NOISE or not NAMEISH.match(name):
        return None
    if item.get("conf", 1.0) < 0.3:
        return None
    return name


def pick_name(texts: list[dict]) -> str | None:
    """Speaker view: choose the name label nearest the bottom-left corner."""
    candidates = []
    for item in texts:
        name = _nameish(item)
        if not name:
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


def _strip_label_ok(name: str) -> bool:
    """Stricter check for tile-strip membership: display names are
    Title Case (or caseless scripts like CJK). Filters out shared-screen
    UI rows that happen to be evenly spaced ("x Deal Status", "Save as
    New") without touching the speaker-view path."""
    return all(not w[0].islower() for w in name.split())


def _is_strip(row: list[tuple]) -> bool:
    """True if these same-row labels are spaced like a run of equal tiles."""
    if len(row) < STRIP_MIN_NAMES:
        return False
    xs = sorted(x for _, x, _, _ in row)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    med = sorted(gaps)[len(gaps) // 2]
    if med <= 0.01:
        return False
    tol = STRIP_GAP_TOL * med
    # a gap may also be ~2× the median: one label in the run OCR didn't read
    return all(abs(g - med) <= tol or abs(g - 2 * med) <= tol for g in gaps)


def find_strips(texts: list[dict]) -> list[list[tuple[str, dict]]]:
    """Rows of ≥4 evenly spaced name labels — the participant tile strip
    Zoom/Teams show in gallery view or alongside a screen share.
    Returns rows of (name, ocr_item), left to right."""
    named = []
    for i, item in enumerate(texts):
        name = _nameish(item)
        if name and _strip_label_ok(name):
            named.append((item.get("y", 0.0), item.get("x", 0.0), i, item, name))
    named.sort(key=lambda e: (e[0], e[1]))

    rows: list[list[tuple]] = []
    row: list[tuple] = []
    for y, x, _, item, name in named:
        if row and y - row[-1][0] > STRIP_Y_TOL:
            rows.append(row)
            row = []
        row.append((y, x, name, item))
    if row:
        rows.append(row)

    # a spreadsheet/CRM table on a shared screen also yields evenly spaced
    # Title Case rows — but many of them, stacked densely (and names may
    # qualify in only some rows). Real tile labels sit a full tile height
    # from any other run of names, so reject a candidate strip whenever
    # another row of ≥3 name-ish texts is right next to it.
    dense = [r for r in rows if len(r) >= 3]
    strips = [r for r in rows if _is_strip(r)
              and not any(d is not r and abs(d[0][0] - r[0][0]) < STRIP_ROW_MIN_DY
                          for d in dense)]
    return [[(name, item) for _, _, name, item in sorted(r, key=lambda e: e[1])]
            for r in strips]


def _glow(item: dict) -> float:
    """Strength of a consistent colored outline around this label's tile."""
    gl = item.get("gl")
    if not gl or len(gl) != 4:
        return 0.0
    left_frac, bottom_frac, left_hue, bottom_hue = (float(v) for v in gl)
    dh = abs(left_hue - bottom_hue) % 360
    if min(dh, 360 - dh) > GLOW_HUE_TOL:
        return 0.0  # two different colors = video content, not one outline
    return min(left_frac, bottom_frac)


def pick_from_strips(strips: list[list[tuple[str, dict]]]) -> str | None:
    """The strip tile whose outline clearly glows — else abstain."""
    scored = sorted(((_glow(item), name) for row in strips for name, item in row),
                    reverse=True)
    if not scored or scored[0][0] < GLOW_MIN:
        return None
    if len(scored) > 1 and scored[0][0] < GLOW_DOMINANCE * max(scored[1][0], 0.04):
        return None
    return scored[0][1]


def analyze(texts: list[dict]) -> tuple[str | None, list[str]]:
    """One OCR frame → (active speaker or None, roster names in tile strips).

    When a tile strip is on screen (screen share / gallery), the window's
    bottom-left is shared content, not a name label — so only the strip's
    glow evidence may name the speaker there.
    """
    strips = find_strips(texts)
    if strips:
        roster = [name for row in strips for name, _ in row]
        return pick_from_strips(strips), roster
    return pick_name(texts), []


class SpeakerTracker:
    """Runs the OCR helper and keeps a rolling (time, name) timeline."""

    def __init__(self):
        self.samples: deque = deque(maxlen=1800)  # ~30 min at 1 Hz
        self.proc: subprocess.Popen | None = None
        self._roster: Counter = Counter()
        self._strip_frames = 0
        self._stopping = False

    def start(self):
        if self.proc:
            return
        if not SPEAKER_BINARY.exists():
            raise RuntimeError(f"speaker helper missing: {SPEAKER_BINARY} — run setup.sh")
        self._stopping = False
        self.proc = subprocess.Popen(
            [str(SPEAKER_BINARY)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # the helper dies immediately when Screen Recording permission is
        # missing — surface that instead of silently showing no names
        time.sleep(0.8)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace").strip()
            self.proc = None
            raise RuntimeError(
                err.replace("[speaker] ", "").replace("\n", " ")
                or "speaker helper exited at startup")
        for target in (self._read_frames, self._read_stderr):
            threading.Thread(target=target, daemon=True).start()
        log.info("speaker OCR running (experimental)")

    def _ingest(self, frame: dict):
        speaker, strip_names = analyze(frame.get("texts", []))
        now = float(frame.get("time") or time.time())
        if strip_names:
            self._strip_frames += 1
            self._roster.update(set(strip_names))
        if speaker:
            self._roster[speaker] += 1
            self.samples.append((now, speaker))

    def _read_frames(self):
        proc = self.proc
        assert proc and proc.stdout
        for line in proc.stdout:
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            self._ingest(frame)
        if not self._stopping:  # EOF without stop(): the helper died
            log.warning("speaker helper exited unexpectedly — captions will "
                        "show 'Participant' until Names is toggled off/on")

    def _read_stderr(self):
        proc = self.proc
        assert proc and proc.stderr
        for line in proc.stderr:
            log.info("speaker: %s", line.decode(errors="replace").rstrip())

    LEAD_GUARD_S = 1.5  # previous speaker's outline lingers this long into a turn
    TRAIL_S = 2.0       # this speaker's outline appears late / persists past the words
    CARRY_S = 10.0      # how far back a name may be carried when nothing voted

    def name_for(self, t0: float, t1: float) -> str | None:
        """Who was speaking during the caption window [t0, t1]?

        The glow evidence lags the audio: meeting apps draw the outline a
        beat after someone starts talking and keep it lit for a second or two
        after they stop. So samples at the start of the window often still
        show the *previous* speaker's fading outline — counting them mislabels
        every turn change — while samples near and just past the end are the
        most trustworthy. Votes therefore skip the first LEAD_GUARD_S of the
        window, and samples up to TRAIL_S after it count at half weight (they
        could already belong to the next speaker's reply). Ties break toward
        the name seen latest.

        The signal also flickers mid-turn (the outline fades between
        sentences, OCR misses a frame), so an empty vote set falls back to
        the most recent name within CARRY_S: speakers keep the floor for many
        seconds at a time, so the nearest recent name is far more often right
        than no name at all.
        """
        votes: dict[str, list[float]] = {}  # name -> [weight, latest sample time]
        for t, name in list(self.samples):
            if t0 + self.LEAD_GUARD_S <= t <= t1:
                w = 2.0
            elif t1 < t <= t1 + self.TRAIL_S:
                w = 1.0
            else:
                continue
            rec = votes.setdefault(name, [0.0, t])
            rec[0] += w
            rec[1] = max(rec[1], t)
        if votes:
            return max(votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        for t, name in reversed(list(self.samples)):
            if t > t1 + self.TRAIL_S:
                continue
            if t < t0 - self.CARRY_S:
                break
            return name
        return None

    def roster(self) -> list[str]:
        """Attendee names seen reliably on screen, most frequent first.
        A name must persist across frames — one misread never enters."""
        min_frames = max(3, self._strip_frames // 10)
        return [n for n, c in self._roster.most_common(40) if c >= min_frames][:30]

    def stop(self):
        if self.proc:
            self._stopping = True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


def _debug(seconds: float = 30.0):  # pragma: no cover
    """`python -m scribatim.speaker [seconds]` — run during a live meeting
    while someone is talking. Dumps what the OCR sees, what the picker
    chooses, and finishes with a self-graded verdict on whether speaker
    naming works for this meeting's layout.
    Run from a terminal that has the Screen Recording permission."""
    from collections import Counter as _Counter
    from pathlib import Path
    out_path = Path("~/Documents/Scribatim/speaker_debug.txt").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    proc = subprocess.Popen(
        [str(SPEAKER_BINARY)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    threading.Thread(
        target=lambda: [emit(f"  [helper] {l.decode(errors='replace').rstrip()}")
                        for l in proc.stderr], daemon=True).start()
    deadline = time.time() + seconds
    frames = strip_frames = glow_frames = 0
    speakers: _Counter = _Counter()
    roster_seen: _Counter = _Counter()
    try:
        for line in proc.stdout:
            if time.time() > deadline:
                break
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            frames += 1
            texts = frame.get("texts", [])
            emit(f"--- frame {frames}: {len(texts)} texts ---")
            for it in sorted(texts, key=lambda i: (-i.get('y', 0), i.get('x', 0))):
                gl = it.get("gl")
                glow = f" gl={gl}" if gl else ""
                emit(f"  x={it.get('x', 0):.2f} y={it.get('y', 0):.2f} "
                     f"conf={it.get('conf', 0):.2f}{glow}  {it.get('text', '')!r}")
            speaker, roster = analyze(texts)
            emit(f"  => speaker: {speaker!r}   strip roster: {roster}")
            if roster:
                strip_frames += 1
                roster_seen.update(set(roster))
                if any(_glow(item) > 0 for row in find_strips(texts) for _, item in row):
                    glow_frames += 1
            if speaker:
                speakers[speaker] += 1
    finally:
        proc.terminate()

    emit("\n===== verdict =====")
    if not frames:
        emit("✗ no frames captured — is a Zoom/Teams meeting window open, and does "
             "this terminal have the Screen Recording permission?")
    else:
        emit(f"frames: {frames}   with tile strip: {strip_frames}   "
             f"with glow signal: {glow_frames}")
        if roster_seen:
            emit(f"roster: {[n for n, _ in roster_seen.most_common(30)]}")
        if speakers:
            emit(f"✓ speaker named in {sum(speakers.values())}/{frames} frames: "
                 f"{dict(speakers.most_common())}")
            emit("  → if these names match who was actually talking, captions will "
                 "be attributed correctly. Done.")
        elif strip_frames and not glow_frames:
            emit("✗ tile strip found but no glow on any tile. If someone WAS "
                 "talking on camera, the outline sampling missed it — share "
                 "this file so the glow thresholds can be tuned.")
        elif strip_frames:
            emit("~ glow present but no clear single winner — thresholds may be "
                 "too strict for this layout; share this file to tune them.")
        else:
            emit("✗ no tile strip and no speaker-view label recognized — share "
                 "this file so the layout heuristics can be extended.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    _debug(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
