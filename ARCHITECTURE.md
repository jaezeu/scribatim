# Architecture & design decisions

One page on the *why* behind choices that look arbitrary from the code alone.
The user-facing behavior is in the [README](README.md); the data flow is in
[docs/pipeline.svg](docs/pipeline.svg).

## Process model

One Python server (`scribatim/`) plus up to two Swift subprocess helpers
(`capture/`, compiled to `bin/` by `setup.sh`):

| Helper | What | Protocol on stdout |
|---|---|---|
| `scribatim-tap` | system-audio via Core Audio process tap | JSON header line, then raw f32 mono |
| `scribatim-speaker` | meeting-window OCR (optional) | one JSON line per frame |

(The mic is captured in-process via sounddevice/PortAudio — see below for
why it deliberately is *not* a voice-processing helper.)

**Why subprocesses instead of in-process bindings:** the capture APIs are
Swift/ObjC-only; a wedged or crashed helper can't take down the server (Core
Audio aggregate devices *do* occasionally wedge); and each helper maps to
exactly one macOS privacy permission, so a user can grant/deny them
independently. The pipe protocol is deliberately trivial — a JSON header or
JSON lines — so helpers stay debuggable with `./bin/scribatim-tap | head`.

**Graceful degradation everywhere:** the mic falls back from 16 kHz to the
device's native rate, MLX falls back to CPU, speaker OCR failing just means
unnamed captions. A meeting must never be lost because an optional layer
broke.

## Capture decisions

- **System audio** uses Core Audio process taps (macOS 14.4+) rather than a
  virtual output device (BlackHole-style): nothing to install system-wide,
  no audio-path changes the user can hear, works with any app.
- **Mic: raw capture, no Apple voice processing.** An earlier version
  captured the mic through the system's voice-processing unit (AEC, so an
  open-speaker mic wouldn't hear the meeting playback). It was removed after
  measurement: while any process has voice processing engaged on a device,
  macOS hands every *other* client of that device — Teams, Zoom — a signal
  attenuated by ~40 dB (a concurrent raw capture drops to ~1% amplitude and
  recovers the moment the unit disengages; the hardware input volume never
  moves, so guarding/restoring it does not help). In short: AEC here made
  the user inaudible in their own meeting. The mic is therefore read raw
  from the default input device, and the echo problem AEC solved is handled
  in Python instead by an **EchoGate** (`audio.py`): the system tap is a
  clean reference of exactly what the speakers played, so a mic segment
  whose normalized cross-correlation against the time-aligned tap audio
  peaks high (bleed measures ~0.9, unrelated speech ~0.03, threshold 0.30)
  is speaker bleed and is dropped — its content is already being transcribed
  from the system lane. The check runs on the transcriber worker (via its
  `prefilter` hook), not in the PortAudio callback that closes segments: the
  correlation FFT on a long segment is too heavy for the real-time audio
  thread. On a headset there is no acoustic path and the gate never fires;
  either way nothing system-wide is touched.
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
  or bare interpolation: a 48 kHz stream decimated without a low-pass folds
  the 8–12 kHz band onto speech frequencies — inaudible in a meter,
  measurable in word accuracy.
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
dominated [segment start, segment end]?" — a windowed vote that absorbs
Whisper's multi-second decode lag and single-frame OCR flicker without any
clock coupling between the helpers. The vote is *skewed late*, because the
glow evidence lags the audio: apps draw the outline a beat after someone
starts talking and keep it lit a second or two after they stop. Samples in
the first ~1.5 s of a window usually still show the previous speaker's
fading outline — counting them mislabeled every turn change — so they're
skipped; samples up to 2 s past the window count at half weight (they catch
outlines drawn after a short utterance, but might already be the next
speaker's reply). An empty vote falls back to the most recent name within
the last 10 s: speakers hold the floor for many seconds at a time, so the
nearest recent name beats an anonymous "Participant" far more often than it
mislabels a fast hand-off.

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
