# Architecture & design decisions

One page on the *why* behind choices that look arbitrary from the code alone.
The user-facing behavior is in the [README](README.md); the data flow is in
[docs/pipeline.svg](docs/pipeline.svg).

## Process model

One Python server (`susurro/`) plus up to three Swift subprocess helpers
(`capture/`, compiled to `bin/` by `setup.sh`):

| Helper | What | Protocol on stdout |
|---|---|---|
| `susurro-tap` | system-audio via Core Audio process tap | JSON header line, then raw f32 mono |
| `susurro-mic` | mic via Apple voice processing (AEC) | same |
| `susurro-speaker` | meeting-window OCR (optional) | one JSON line per frame |

**Why subprocesses instead of in-process bindings:** the capture APIs are
Swift/ObjC-only; a wedged or crashed helper can't take down the server (Core
Audio aggregate devices *do* occasionally wedge); and each helper maps to
exactly one macOS privacy permission, so a user can grant/deny them
independently. The pipe protocol is deliberately trivial — a JSON header or
JSON lines — so helpers stay debuggable with `./bin/susurro-tap | head`.

**Graceful degradation everywhere:** AEC mic falls back to a raw sounddevice
stream, MLX falls back to CPU, speaker OCR failing just means unnamed
captions. A meeting must never be lost because an optional layer broke.

## Capture decisions

- **System audio** uses Core Audio process taps (macOS 14.4+) rather than a
  virtual output device (BlackHole-style): nothing to install system-wide,
  no audio-path changes the user can hear, works with any app.
- **Mic AEC** (`susurro-mic`) exists because the mic otherwise hears the
  meeting playing from open speakers, and Whisper transcribes whoever is
  loudest — the user's own lane gets drowned. Two non-obvious details:
  ducking by the voice unit is explicitly disabled (it would quiet the very
  meeting audio the tap is transcribing), and the helper streams the
  *strongest* channel instead of averaging — the voice unit reports phantom
  channels (9 ch for a 1-ch laptop mic), and averaging silent padding would
  attenuate the voice up to 9×, below the speech gate.
- **Speaker OCR** (`susurro-speaker`) reads the name the meeting app already
  draws on screen — attribution without a bot in the call. The Swift side is
  deliberately dumb (emit all recognized text + positions); *which* text is
  the name is decided in Python (`speaker.py`), because the heuristics are
  the brittle part and iterating on them must not require recompiling.
  Window selection rejects Teams' main Chat/Activity windows by title —
  capturing the wrong window is worse than none, since chat sender names
  also sit bottom-left and would *mislabel* speakers. In gallery view (>3
  plausible names) the picker abstains rather than guesses.

## Transcription

- Utterances are cut by an adaptive energy gate (`audio.py:Segmenter`), not
  by Whisper's VAD, so segmentation cost is near-zero and per-utterance
  latency is bounded by `segment_silence_seconds`.
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

OCR samples land on a rolling timeline; each caption asks "which name
dominated [segment start, segment end]?" — a windowed majority vote. This
absorbs Whisper's multi-second decode lag and single-frame OCR flicker
without any clock coupling between the helpers.

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
artifacts on disk are text files under `~/Documents/Susurro` (mode `0700`).

## Config philosophy

`config.json` is the shipped template — neutral defaults, every key present
so users discover the knobs. Personalization (vocabulary, meeting context)
stays local and uncommitted. Anything experimental (speaker OCR) defaults
off so no privacy permission is requested until the user opts in.
