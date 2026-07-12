import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "whisper_model": "medium",
    "whisper_backend": "auto",  # auto | mlx (Metal GPU) | ct2 (CPU)
    "language": "",  # "" = auto-detect per utterance; or pin e.g. "ja", "zh", "ko"
    "vocabulary": "",
    "meeting_context": "",
    "compute_type": "int8",
    "beam_size": 1,
    "show_original": True,
    "ollama_model": "llama3.2:3b",
    "ollama_url": "http://127.0.0.1:11434",
    "llm_num_ctx": 8192,  # LLM context window; long meetings are chunked to fit
    "port": 8710,
    "save_dir": "~/Documents/Scribatim",
    "segment_max_seconds": 12.0,
    "segment_silence_seconds": 0.8,
    "segment_min_speech_seconds": 0.4,
    "mic_aec": True,  # echo-cancelled mic via Apple voice processing
    "speaker_ocr": False,  # experimental: name captions via meeting-window OCR
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = PROJECT_ROOT / "config.json"
    if path.exists():
        try:
            user = json.loads(path.read_text())
        except ValueError as e:
            raise SystemExit(f"✗ {path} is not valid JSON: {e}") from e
        # a typo'd key silently doing nothing is worse than a warning
        unknown = sorted(set(user) - set(DEFAULTS))
        if unknown:
            print(f"config.json: ignoring unknown key(s): {', '.join(unknown)} "
                  f"— see DEFAULTS in scribatim/config.py", file=sys.stderr)
        cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    cfg["save_dir"] = str(Path(cfg["save_dir"]).expanduser())
    return cfg
