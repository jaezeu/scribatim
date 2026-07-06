# Susurro

*Susurro — Spanish for "whisper".*

A fully-local meeting copilot for macOS. It listens to **system audio** (whatever
Teams / Zoom / Meet / a browser tab is playing) plus your **microphone**, shows
**live captions translated to English** from any language — including speakers
who **mix languages mid-sentence** — and generates **meeting minutes, quick
recaps, and follow-up email drafts** with a local LLM.

**No cloud. No bots joining your call. No audio files on disk. Nothing leaves your Mac.**

```
 Teams/Zoom/Meet ──▶ macOS audio ──▶ Core Audio tap (Swift) ─┐
                                                             ├─▶ Whisper (local) ─▶ live captions (EN)
 Your voice ───────▶ microphone ─────────────────────────────┘        │
                                                                      ▼
                                                     Ollama llama3.2 (local) ─▶ minutes.md
```

## Requirements

- macOS **14.4+** (Apple Silicon or Intel) — uses Core Audio process taps
- Xcode Command Line Tools (`xcode-select --install`)
- [Homebrew](https://brew.sh) and Python **3.10+**
- ~3 GB disk for the local models (Whisper `small` + `llama3.2:3b`)

## Quick start

```bash
git clone <this-repo> && cd <this-repo>
./setup.sh        # one-time: compiles the audio tap, installs deps, downloads models
./run.sh          # prints a private dashboard URL like http://127.0.0.1:8710/?t=<token>
```

1. Open the URL, press **▶ Start** before (or during) your meeting.
2. Captions stream in two lanes — *Participants* (system audio, auto-translated,
   original text underneath) and *You* (mic). Language is detected per utterance,
   and code-switching within an utterance is handled.
3. Three one-click deliverables, all generated on-device:
   **⚡ Recap** (mid-meeting "where are we"), **✦ Minutes** (summary, decisions,
   action-item table with owners), **✉ Email** (ready-to-send follow-up draft).
   **Copy** puts the Markdown on your clipboard.
4. Everything is also saved to `~/Documents/Susurro/<date_time>/`
   (`transcript.md`, `minutes.md`, `summary.md`, `email.md`).

First launch triggers two one-time macOS privacy prompts for your terminal:
**System Audio Recording** and **Microphone**. If you miss one, re-enable under
*System Settings → Privacy & Security*.

## Security model

- Web server binds `127.0.0.1` only — unreachable from your network.
- Every request requires a random per-launch token (in the printed URL, then a
  `HttpOnly`/`SameSite=Strict` cookie). Other local processes can't snoop the UI.
- Raw audio lives only in memory; it is never written to disk.
- Whisper and the minutes LLM (Ollama) run entirely on-device. The only network
  activity ever is the **one-time model download** during `setup.sh`.
- Saved transcripts/minutes go to `~/Documents/Susurro` (mode `0700`).

> ⚖️ Recording/transcribing calls may require participant consent depending on
> your jurisdiction and company policy — that decision is yours.

## Tuning (`config.json`)

| Key | Default | Notes |
|-----|---------|-------|
| `whisper_model` | `small` | `medium` / `large-v3` = better accuracy, more latency |
| `language` | `""` (auto) | pin the source language (e.g. `"ja"`, `"zh"`, `"ko"`, `"hi"`) if auto-detect misfires |
| `show_original` | `true`  | also show untranslated text under each caption |
| `ollama_model`  | `llama3.2:3b` | any Ollama model, e.g. `qwen2.5:7b` for richer minutes |
| `vocabulary` | `""` | jargon/product names to bias transcription, e.g. `"Kubernetes, Terraform, POC"` |
| `meeting_context` | `""` | one sentence about you/your meetings to shape minutes & emails, e.g. `"You are a sales engineer meeting customers."` |
| `segment_silence_seconds` | `0.8` | pause length that ends an utterance |

After changing `whisper_model` or `ollama_model`, run `./setup.sh` once to fetch it.

### Asian & other non-Latin languages

All Whisper languages work out of the box — Chinese, Cantonese, Japanese,
Korean, Thai, Vietnamese, Hindi, Tamil, Indonesian, Tagalog, and ~90 more.
Original-language captions render with the correct script, glyph variants,
and line-breaking. Two tips for best results:

- The `small` model's accuracy drops noticeably on Asian languages; set
  `whisper_model` to `medium` (good balance) or `large-v3` (best) for
  CJK/Thai/Vietnamese-heavy meetings, then re-run `./setup.sh`.
- Auto-detection can confuse related languages on short utterances
  (e.g. Mandarin/Cantonese/Japanese, Malay/Indonesian). For a
  single-language meeting, pin it with e.g. `"language": "ja"` — this also
  skips detection, reducing latency. Leave it `""` for mixed-language calls.

## Platform support

macOS only, by design: the app-agnostic capture relies on Core Audio process
taps, which have no direct Windows/Linux equivalent. Everything except
`capture/systemaudio.swift` is portable Python — a Windows port would swap in a
WASAPI-loopback capture backend and keep the rest unchanged. PRs welcome.

## License & third-party

- This project: [MIT](LICENSE).
- Python dependencies (FastAPI, uvicorn, faster-whisper, sounddevice, NumPy,
  requests) are MIT/BSD/Apache-2.0 licensed.
- Models are **downloaded by you at setup time, not distributed with this repo**:
  Whisper weights are MIT; the default minutes model (`llama3.2:3b`) is covered
  by the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
  — swap `ollama_model` for e.g. `qwen2.5:7b` (Apache-2.0) if that matters to you.

## Notes & limits

- Works with **any** meeting app — nothing is injected into the call; the tap is
  read-only on the Mac's output mix.
- If you switch audio output devices mid-meeting (e.g. AirPods connect), press
  Stop/Start to re-attach the tap to the new device.
- Translation quality: Whisper translates *to English only* (that's the use case).
- `small` runs comfortably in real time on Apple Silicon; bump to `medium` if
  accuracy on heavy accents matters more than latency.
