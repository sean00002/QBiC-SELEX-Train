"""Configuration: every hyperparameter lives in config/config.yaml."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Paths in the config are relative to the repository root.
    for section, key in (("bias_model", "path"), ("data", "manifest"), ("data", "reads_dir")):
        value = cfg.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(REPO_ROOT / value)
    return cfg
