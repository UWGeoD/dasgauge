#!/usr/bin/env python3
"""Split a configured continuous DAS recording into paired sample arrays."""

import argparse
import json
from pathlib import Path

from dasgauge.config import find_dataset_config, preprocess_steps
from dasgauge.io import DASRecording
from dasgauge.sampling import make_run_name, sampling_preview, split_recording


REPO_DIR = Path(__file__).resolve().parent


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="hold_skip", choices=("hold_skip",))
    parser.add_argument("--s1", type=float, help="seconds retained in each sample")
    parser.add_argument("--s2", type=float, help="seconds skipped after each sample")
    parser.add_argument("--dataset", default="newville_nov", help="dataset_id or YAML path")
    parser.add_argument(
        "--coupling-channels",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="zero-based half-open coupling range",
    )
    parser.add_argument(
        "--uncoupling-channels",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="zero-based half-open uncoupling range",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_DIR / "samples")
    parser.add_argument("--dry-run", action="store_true", help="validate and estimate without writing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing parameter-identical run after staging the new output",
    )
    return parser


def _setting(cli_value, defaults, key):
    value = cli_value if cli_value is not None else defaults.get(key)
    if value is None:
        raise ValueError(f"Set --{key.replace('_', '-')} or define sampling_defaults.{key}")
    return value


def main(argv=None):
    args = _parser().parse_args(argv)
    config = find_dataset_config(args.dataset, REPO_DIR / "configs")
    root = Path(config["recording_dir"])
    files = list(root.glob(config.get("file_pattern", "*.h5")))
    recording = DASRecording(files, vendor=config["vendor"])

    defaults = config.get("sampling_defaults", {})
    strategy_defaults = defaults.get("strategies", {}).get(args.strategy, {})
    parameters = {
        "s1": _setting(args.s1, strategy_defaults, "s1"),
        "s2": _setting(args.s2, strategy_defaults, "s2"),
    }
    coupling = tuple(_setting(args.coupling_channels, defaults, "coupling_channels"))
    uncoupling = tuple(_setting(args.uncoupling_channels, defaults, "uncoupling_channels"))
    steps = preprocess_steps(config)

    preview = sampling_preview(
        recording,
        strategy=args.strategy,
        parameters=parameters,
        coupling_channels=coupling,
        uncoupling_channels=uncoupling,
    )
    preview.pop("windows")
    run_name = make_run_name(args.strategy, parameters, coupling, uncoupling, steps)
    output_dir = args.output_root / config["dataset_id"] / args.strategy / run_name
    report = {
        "dataset_id": config["dataset_id"],
        "source_file_count": len(recording.parts),
        "source_files_temporally_ordered": [part.path.name for part in recording.parts],
        "channels": {"coupling": coupling, "uncoupling": uncoupling},
        "strategy": {"name": args.strategy, "parameters": parameters},
        "preprocess": [{"name": name, "params": params} for name, params in steps],
        "output_dir": str(output_dir),
        "preview": preview,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    manifest = split_recording(
        recording,
        output_dir,
        dataset_id=config["dataset_id"],
        strategy=args.strategy,
        parameters=parameters,
        coupling_channels=coupling,
        uncoupling_channels=uncoupling,
        preprocess_steps=steps,
        config_path=config["_config_path"],
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_dir": str(output_dir), "samples": manifest["samples"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
