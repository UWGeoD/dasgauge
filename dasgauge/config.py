"""Dataset configuration loading and validation."""

import os
from pathlib import Path

import yaml

__all__ = ["find_dataset_config", "load_dataset_config", "preprocess_steps"]


def load_dataset_config(path, *, expand_recording_dir=True):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    required = ("dataset_id", "recording_dir", "vendor")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Dataset config {path} is missing: {', '.join(missing)}")
    if expand_recording_dir:
        recording_dir = os.path.expanduser(
            os.path.expandvars(str(config["recording_dir"]))
        )
        if "$" in recording_dir:
            raise ValueError(
                f"Dataset config {path} has an unresolved recording_dir environment variable"
            )
        config["recording_dir"] = recording_dir
    config["_config_path"] = str(path.resolve())
    return config


def find_dataset_config(dataset, config_dir="configs"):
    """Resolve a config path or a ``dataset_id`` within ``config_dir``."""
    candidate = Path(dataset)
    if candidate.is_file():
        return load_dataset_config(candidate)

    matches = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        try:
            config = load_dataset_config(path, expand_recording_dir=False)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if config.get("dataset_id") == dataset:
            matches.append(config)
    if not matches:
        raise FileNotFoundError(
            f"No dataset config with dataset_id={dataset!r} in {config_dir}"
        )
    if len(matches) > 1:
        paths = [item["_config_path"] for item in matches]
        raise ValueError(f"Multiple configs define dataset_id={dataset!r}: {paths}")
    return load_dataset_config(matches[0]["_config_path"])


def preprocess_steps(config):
    """Convert YAML preprocessing entries into ``make_preprocess`` tuples."""
    raw_steps = config.get("preprocess", [])
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_steps, list):
        raise ValueError("preprocess must be a list (use [] for no preprocessing)")

    steps = []
    for index, entry in enumerate(raw_steps):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"preprocess[{index}] must contain a name")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"preprocess[{index}].params must be a mapping")
        steps.append((str(entry["name"]), dict(params)))
    return steps
