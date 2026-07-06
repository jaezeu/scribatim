import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "whisper_model": "medium",
    "language": "",  # "" = auto-detect per utterance; or pin e.g. "ja", "zh", "ko"
    "vocabulary": "",
    "meeting_context": "",
    "compute_type": "int8",
    "beam_size": 1,
    "show_original": True,
    "ollama_model": "llama3.2:3b",
    "ollama_url": "http://127.0.0.1:11434",
    "port": 8710,
    "save_dir": "~/Documents/Susurro",
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
        cfg.update(json.loads(path.read_text()))
    cfg["save_dir"] = str(Path(cfg["save_dir"]).expanduser())
    return cfg
