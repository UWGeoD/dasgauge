"""Tests for the (s1, s2) x comparison sweep orchestrator."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sweep_c2st


CONFIG = """
dataset_id: sweep_test
recording_dir: /nonexistent
vendor: optasense
preprocess: []
sampling_defaults:
  coupling_channels: [10, 20]
  uncoupling_channels: [30, 40]
  strategies:
    hold_skip: {s1: 5, s2: 5}
"""


def _write_result(
    run_dir, args, comparison, *, timestamp, rejection_rate, n_pairs, method=None
):
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / (
        sweep_c2st._result_prefix(args, comparison, method) + timestamp + ".json"
    )
    summary = {
        "repeat_count": args.repeats,
        "alpha": args.alpha,
        "rejection_count": int(rejection_rate * args.repeats),
        "rejection_rate": rejection_rate,
        "accuracy_mean": 0.6,
        "accuracy_median": 0.61,
        "accuracy_minimum": 0.4,
        "accuracy_maximum": 0.8,
        "p_value_mean": 0.2,
        "p_value_median": 0.15,
        "p_value_minimum": 0.001,
        "p_value_maximum": 0.9,
    }
    trial = {
        "grouping": {"group_a_sample_indices": list(range(n_pairs))},
        "c2st": {"k": 7, "n_test_examples": n_pairs},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": method,
                "target_null": "test-null" if method else None,
                "summary": summary,
                "trials": [trial],
            },
            handle,
        )
    return path


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config_dir = self.root / "configs"
        config_dir.mkdir()
        self.config_path = config_dir / "sweep_test.yaml"
        self.config_path.write_text(CONFIG, encoding="utf-8")
        self.output_root = self.root / "samples"

    def _args(self, *extra):
        return sweep_c2st._parser().parse_args(
            [
                "--dataset", str(self.config_path),
                "--output-root", str(self.output_root),
                "--s1", "5", "10",
                "--s2", "5", "20",
                *extra,
            ]
        )

    def test_plan_covers_the_grid_with_split_samples_run_names(self):
        args = self._args()
        config, coupling, uncoupling, cells = sweep_c2st._plan(args)
        self.assertEqual(config["dataset_id"], "sweep_test")
        self.assertEqual((coupling, uncoupling), ((10, 20), (30, 40)))
        self.assertEqual([(cell.s1, cell.s2) for cell in cells], [(5, 5), (5, 20), (10, 5), (10, 20)])
        self.assertEqual(
            cells[0].run_dir,
            self.output_root / "sweep_test" / "hold_skip"
            / "hold_skip__s1-5__s2-5__c-10-20__u-30-40__pp-none",
        )
        self.assertEqual(len({cell.run_dir for cell in cells}), 4)

    def test_cli_channels_override_the_config_defaults(self):
        args = self._args("--coupling-channels", "1", "2", "--uncoupling-channels", "3", "4")
        _, coupling, uncoupling, cells = sweep_c2st._plan(args)
        self.assertEqual((coupling, uncoupling), ((1, 2), (3, 4)))
        self.assertIn("c-1-2__u-3-4", cells[0].run_dir.name)

    def test_existing_result_matches_configuration_and_takes_the_newest(self):
        args = self._args()
        _, _, _, cells = sweep_c2st._plan(args)
        run_dir = cells[0].run_dir
        _write_result(run_dir, args, "s-p", timestamp="20260101T000000Z", rejection_rate=0.1, n_pairs=8)
        newest = _write_result(
            run_dir, args, "s-p", timestamp="20260102T000000Z", rejection_rate=0.2, n_pairs=8
        )
        self.assertEqual(sweep_c2st._existing_result(run_dir, args, "s-p"), newest)
        self.assertIsNone(sweep_c2st._existing_result(run_dir, args, "s-s"))

        other = self._args("--repeats", "7")
        self.assertIsNone(sweep_c2st._existing_result(run_dir, other, "s-p"))

    def test_collect_writes_csv_json_and_markdown_grids(self):
        args = self._args()
        _, _, _, cells = sweep_c2st._plan(args)
        for index, cell in enumerate(cells):
            cell.run_dir.mkdir(parents=True)
            with (cell.run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump({"samples": {"count": 100 - index}}, handle)
            for comparison in args.comparisons:
                _write_result(
                    cell.run_dir,
                    args,
                    comparison,
                    timestamp="20260101T000000Z",
                    rejection_rate=0.5,
                    n_pairs=50 - index,
                )

        sweep_dir = self.root / "sweep"
        rows = sweep_c2st.stage_collect(args, cells, sweep_dir)
        self.assertEqual(len(rows), 4 * len(args.comparisons))

        lines = (sweep_dir / "summary.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(","), sweep_c2st.CSV_FIELDS)
        self.assertEqual(len(lines), 1 + 4 * len(args.comparisons))

        payload = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["grid"], {"s1": [5, 10], "s2": [5, 20]})
        self.assertEqual(payload["rows"][0]["n_samples"], 100)
        self.assertEqual(payload["rows"][0]["n_pairs"], 50)
        self.assertFalse(Path(payload["rows"][0]["result_file"]).is_absolute())
        self.assertNotIn("created_at_utc", payload)

        markdown = (sweep_dir / "summary.md").read_text(encoding="utf-8")
        for comparison in args.comparisons:
            self.assertIn(f"## {comparison}", markdown)
        self.assertIn("| **5 s** | 100 | 99 |", markdown)

    def test_collect_tolerates_missing_cells(self):
        args = self._args()
        _, _, _, cells = sweep_c2st._plan(args)
        cells[0].run_dir.mkdir(parents=True)
        _write_result(cells[0].run_dir, args, "s-p", timestamp="20260101T000000Z",
                      rejection_rate=1.0, n_pairs=4)
        rows = sweep_c2st.stage_collect(args, cells, self.root / "sweep")
        self.assertEqual(len(rows), 1)
        self.assertIn("—", (self.root / "sweep" / "summary.md").read_text(encoding="utf-8"))

    def test_collects_multiple_explicit_methods(self):
        methods = ["original_binomial_c2st", "paired_mcnemar"]
        args = self._args("--methods", *methods)
        _, _, _, cells = sweep_c2st._plan(args)
        cell = cells[0]
        cell.run_dir.mkdir(parents=True)
        for method in methods:
            _write_result(
                cell.run_dir,
                args,
                "s-p",
                method=method,
                timestamp="20260101T000000Z",
                rejection_rate=0.5,
                n_pairs=8,
            )
        rows = sweep_c2st.stage_collect(args, cells, self.root / "multi")
        self.assertEqual({row["method"] for row in rows}, set(methods))
        payload = json.loads((self.root / "multi" / "summary.json").read_text())
        self.assertEqual(payload["methods"], methods)


if __name__ == "__main__":
    unittest.main()
