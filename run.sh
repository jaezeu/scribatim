#!/bin/zsh
# Launch Susurro (dashboard URL with access token is printed on start).
cd "$(dirname "$0")"

if [[ ! -x bin/susurro-tap || ! -d .venv ]]; then
  echo "First run: executing setup.sh…"
  ./setup.sh || exit 1
fi

# Make sure the local minutes LLM is up (fails soft — captions work without it).
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Starting Ollama service…"
  brew services start ollama >/dev/null 2>&1
fi

exec .venv/bin/python -m susurro.app
