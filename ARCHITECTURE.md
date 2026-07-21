# Architecture & design decisions

One page on the *why* behind choices that look arbitrary from the code alone.
The user-facing behavior is in the [README](README.md); the data flow is in
[docs/pipeline.svg](docs/pipeline.svg).

## Process model

One Python server (`scribatim/`) plus up to three Swift subprocess helpers
(`capture/`, compiled to `bin/` by `setup.sh`):

| Helper | What | Protocol on stdout |
|---|---|---|
| `scribatim-tap` | system-audio via Core Audio process tap | JSON header line, then raw f32 mono |
| `scribatim-mic` | mic via Apple voice processing (AEC) | same |
| `scribatim-speaker` | meeting-window OCR (optional) | one JSON line per frame |

**Why subprocesses instead of in-process bindings:** the capture APIs are
Swift/ObjC-only; a wedged or crashed helper can't take down the server (Core
Audio aggregate devices *do* occasionally wedge); and each helper maps to
exactly one macOS privacy permission, so a user can grant/deny them
independently. The pipe protocol is deliberately trivial — a JSON header or
JSON lines — so helpers stay debuggable with `./bin/scribatim-tap | head`.

**Graceful degradation everywhere:** AEC mic falls back to a raw sounddevice
stream, MLX falls back to CPU, speaker OCR failing just means unnamed
captions. A meeting must never be lost because an optional layer broke.

## Capture decisions

- **System audio** uses Core Audio process taps (macOS 14.4+) rather than a
  virtual output device (BlackHole-style): nothing to install system-wide,
  no audio-path changes the user can hear, works with any app.
- **Mic AEC** (`scribatim-mic`) exists because the mic otherwise hears the
  meeting playing from open speakers, and Whisper transcribes whoever is
  loudest — the user's own lane gets drowned. Three non-obvious details:
  ducking by the voice unit is explicitly disabled (it would quiet the very
  meeting audio the tap is transcribing); the helper streams the
  *strongest* channel instead of averaging — the voice unit reports phantom
  channels (9 ch for a 1-ch laptop mic), and averaging silent padding would
  attenuate the voice up to 9×, below the speech gate; and an **input gain
  guard** snapshots the hardware input volume before voice processing is
  enabled and restores it whenever it drops (and again on exit). The
  hardware gain is shared machine-wide, and even with the voice unit's AGC
  disabled macOS winds it down when voice processing engages — Teams/Zoom
  read the same turned-down device, so without the guard the user goes
  quiet *in their own meeting*. Only drops below the snapshot are corrected,
  so a user deliberately raising their mic volume is never fought.
- **Speaker OCR** (`scribatim-speaker`) reads the name the meeting app already
  draws on screen — attribution without a bot in the call. The Swift side is
  deliberately dumb (emit all recognized text + positions, plus a mechanical
  color sample of the bands left of / below each text box); *which* text is
  the name is decided in Python (`speaker.py`), because the heuristics are
  the brittle part and iterating on them must not require recompiling.
  Window selection rejects Teams' main Chat/Activity windows by title —
  capturing the wrong window is worse than none, since chat sender names
  also sit bottom-left and would *mislabel* speakers.

## Transcription

- Utterances are cut by an adaptive energy gate (`audio.py:Segmenter`), not
  by Whisper's VAD, so segmentation cost is near-zero and per-utterance
  latency is bounded by `segment_silence_seconds`.
- Capture rates are converted to Whisper's 16 kHz by a streaming
  **windowed-sinc resampler** (`audio.py:Resampler`), not by box-averaging
  or bare interpolation: the mic helper's 24 kHz stream decimated without a
  low-pass folds the 8–12 kHz band onto speech frequencies — inaudible in a
  meter, measurable in word accuracy.
- Each segment is **peak-normalized** (gain capped at 12×) before decoding —
  Whisper degrades on faint audio (distant speakers on the tap, an AGC-less
  mic) — while hallucination gating below reads the *raw* RMS, so quiet
  audio stays "quiet" to the filter.
- Hallucination defense is layered, because Whisper invents text on
  breaths/echo residue/muted mics — from stock phrases through word loops
  ("demokrat! demokrat!") up to fluent, entirely fabricated sentences:
  * Silero VAD scores every segment's **speech ratio on the raw audio**
    (pre-boost: normalization makes room noise look speech-like to the
    VAD). Zero speech → never decoded. Marginal speech or near-silence →
    decoded in **strict mode**: no gain boost, and any single weak quality
    signal drops a segment.
  * Per-segment drops on Whisper's own quality signals (normal mode:
    `no_speech_prob`>0.6 with `avg_logprob`<-1, or `compression_ratio`>2.4;
    strict mode: any of `no_speech_prob`>0.5, `avg_logprob`<-0.85,
    `compression_ratio`>2.2). Both backends expose these but only use them
    for temperature fallback, not dropping — fabricated sentences carry
    mediocre confidence, which the strict thresholds catch.
  * A text-level gate (`is_hallucination`): stock phrases ("Thank you.")
    and word loops are dropped on quiet audio, extreme loops at any
    volume. Short real repetitions ("No, no, no." said loudly) survive by
    design.
  Relatedly, per-utterance language detection misfires on short English
  clips ("test" → Indonesian); when the original-language pass returns the
  same text as the translation, the caption is relabeled English.
- One worker thread, one queue. Segments are dropped (with a log) if the
  queue fills — stale captions are worse than missing ones in a live tool.
- **Two backends** (`whisper_backend: auto`): MLX on Apple Silicon (Metal
  GPU, ~13× faster than CPU on `medium` — measured), CTranslate2 on CPU as
  the fallback and the Intel path. whisper.cpp + CoreML (the literal Neural
  Engine route) was rejected because its model-conversion step conflicts
  with the one-command `setup.sh`.
- Non-English utterances decode **twice**: translate-to-English for the main
  caption, then transcribe-in-original for the side-by-side text. The
  `vocabulary` prompt is only applied when the *output* is English — priming
  a CJK transcription pass with English text pulls the decoder toward Latin
  output. On the CT2 path, segment joining is script-aware (no spaces
  injected into zh/ja/th/…); MLX inherits Whisper's own concatenation.

## Speaker attribution

Two on-screen layouts, two mechanisms:

- **Speaker view** — the active speaker fills the window; their name label
  is the name-ish text nearest the bottom-left corner.
- **Tile strip / gallery** (incl. during screen shares) — every tile shows
  its name permanently, so text alone can't say who is talking. The strip is
  recognized structurally: ≥4 evenly spaced Title-Case labels on one
  baseline, with no other run of names right next to it (spreadsheets and
  browser tab bars on a shared screen produce name-ish rows too — but
  densely stacked or irregularly spaced, which is how they're rejected;
  all of this was tuned against a real captured Teams screen-share meeting —
  `tests/test_speaker.py` reproduces its geometry with anonymized names).
  The strip yields the **attendee roster**
  (a name must persist across frames to enter), and the active speaker is
  the tile whose surrounding pixels glow in one consistent accent hue — the
  outline meeting apps draw around the speaking tile. No clear glow → no
  guess; captions fall back to "Participant" and minutes still get the
  roster.

OCR samples land on a rolling timeline; each caption asks "which name
dominated [segment start, segment end]?" — a windowed majority vote. This
absorbs Whisper's multi-second decode lag and single-frame OCR flicker
without any clock coupling between the helpers. An empty window (the glow
faded between sentences, OCR missed frames) falls back to the most recent
name within the last 10 s: speakers hold the floor for many seconds at a
time, so the nearest recent name beats an anonymous "Participant" far more
often than it mislabels a fast hand-off.

Audio diarization (pyannote-style) was considered and rejected for v1: it's
a second heavy model competing for compute in real time, and it yields
anonymous "Speaker 1/2" labels, whereas the screen gives real names.

## Minutes (map-reduce)

Ollama **silently drops the front** of a prompt that exceeds `num_ctx` — for
meeting minutes that's the agenda and early decisions, lost without warning.
So `minutes.py` estimates tokens conservatively (~3 chars/token, because
timestamps and names tokenize badly); a transcript over budget is condensed
chunk-by-chunk into dense notes (a prompt that explicitly preserves names,
numbers, dates, commitments), and the deliverable is written from the notes.
Fits-in-one-pass transcripts skip all of this — zero overhead for the
common case.

## Security model

Local-only is a *claim* the design has to enforce: bind `127.0.0.1`, but
also require a per-launch random token, because localhost is reachable by
every process on the machine — a live transcript of a confidential call
deserves a lock even locally. The `Host`-header allowlist blocks DNS
rebinding, the token cookie is `HttpOnly`/`SameSite=Strict`, and CSP pins
everything to `self`. Raw audio and OCR frames live only in memory; the only
artifacts on disk are text files under `~/Documents/Scribatim` (mode `0700`).

## Config philosophy

`config.json` is the shipped template — neutral defaults, every key present
so users discover the knobs. Personalization (vocabulary, meeting context)
stays local and uncommitted. Anything experimental (speaker OCR) defaults
off so no privacy permission is requested until the user opts in.
