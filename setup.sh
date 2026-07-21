#!/bin/zsh
# One-time setup for Scribatim (macOS, Apple Silicon or Intel).
# Everything installed runs locally; the only network use is downloading
# the models once (Whisper + the Ollama minutes model).
set -e
cd "$(dirname "$0")"

fail() { echo "\n✗ $1" >&2; exit 1; }

echo "==> Checking prerequisites…"
[[ "$(uname)" == "Darwin" ]] || fail "Scribatim is macOS-only (needs Core Audio process taps)."
sw_vers -productVersion | awk -F. '{ exit !($1 > 14 || ($1 == 14 && $2 >= 4)) }' \
  || fail "macOS 14.4+ required for system-audio taps (you have $(sw_vers -productVersion))."
command -v swiftc >/dev/null \
  || fail "Swift compiler not found. Install Command Line Tools:  xcode-select --install"
command -v brew >/dev/null \
  || fail "Homebrew not found (needed for Ollama). Install from https://brew.sh"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || fail "Python 3.10+ required (you have $(python3 -V 2>&1))."
echo "    all good"

echo "==> Compiling system-audio tap (Swift)…"
mkdir -p bin
swiftc -O capture/systemaudio.swift -o bin/scribatim-tap \
  -framework CoreAudio -framework AudioToolbox

echo "==> Compiling echo-cancelled mic helper (Swift)…"
swiftc -O capture/micaec.swift -o bin/scribatim-mic \
  -framework AVFoundation -framework CoreAudio

echo "==> Compiling speaker-OCR helper (Swift)…"
swiftc -O capture/speakerocr.swift -o bin/scribatim-speaker \
  -framework ScreenCaptureKit -framework Vision -framework CoreMedia

echo "==> Python environment…"
[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "==> Downloading Whisper model (one-time)…"
.venv/bin/python - <<'PY'
import logging
logging.basicConfig(level=logging.INFO, format="    %(message)s")
from scribatim.config import load_config
from scribatim.transcriber import Transcriber
# loads (and downloads on first run) whichever backend this Mac will use:
# MLX on Apple Silicon, CTranslate2/CPU otherwise
Transcriber(load_config(), lambda e: None).load()
PY

echo "==> Ollama (local minutes LLM)…"
if ! command -v ollama >/dev/null; then
  brew install --quiet ollama
fi
brew services start ollama >/dev/null 2>&1 || true
sleep 2
MODEL=$(python3 -c "import json; print(json.load(open('config.json'))['ollama_model'])")
ollama pull "$MODEL"

echo
echo "✓ Setup complete. Start with:  ./run.sh"
echo "  First launch will ask for two macOS permissions:"
echo "  System Audio Recording + Microphone (System Settings → Privacy & Security)."
