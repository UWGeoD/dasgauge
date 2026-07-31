"""Tests for dasgauge.plotting.

Figures are created with the non-interactive Agg backend and show=False, so no
window is ever opened.
"""

import builtins
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np

import dasgauge.plotting as plotting
from dasgauge.plotting import normalize, plot_das_data, plot_single


class PlottingTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.data = rng.normal(size=(8, 200))
        self.addCleanup(plt.close, "all")


class NormalizeTests(PlottingTestCase):
    def test_zero_mean_unit_std(self):
        y = normalize(self.data)
        np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-9)
        np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-6)


class PlotDasDataTests(PlottingTestCase):
    def test_creates_figure(self):
        fig, ax = plot_das_data(self.data, np.arange(8), dx=2.0, dt=0.01, show=False)

        self.assertIsNotNone(fig)
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(ax.get_xlabel(), "Channel Position (m)")
        self.assertEqual(ax.get_ylabel(), "Time (s)")
        # channel positions scaled by dx; time axis inverted (top = t0)
        self.assertEqual(ax.get_xlim(), (0.0, 14.0))
        self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])

    def test_time_window_and_title(self):
        fig, ax = plot_das_data(
            self.data, np.arange(8), dx=1.0, dt=0.01,
            start_time=0.5, end_time=1.0, title="window", show=False,
        )

        self.assertEqual(ax.get_title(), "window")
        # 0.5-1.0 s at dt=0.01 keeps 51 of 200 samples
        self.assertEqual(ax.images[0].get_array().shape[0], 51)

    def test_draws_into_existing_axes(self):
        fig, axes = plt.subplots(1, 2)

        out_fig, out_ax = plot_das_data(
            self.data, np.arange(8), dx=1.0, dt=0.01, ax=axes[1], show=False
        )

        self.assertIs(out_ax, axes[1])
        self.assertIs(out_fig, fig)
        self.assertEqual(len(axes[0].images), 0)


class PlotSingleTests(PlottingTestCase):
    def test_creates_figure(self):
        fig, ax = plot_single(self.data, 3, dx=2.5, dt=0.01, show=False)

        self.assertGreaterEqual(len(ax.lines), 1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), self.data[3])
        self.assertEqual(ax.get_xlabel(), "Time (s)")
        self.assertEqual(ax.get_ylabel(), "Amplitude")
        self.assertIn("Channel 3", ax.get_title())
        self.assertIn("7.50 m", ax.get_title())

    def test_norm_ref_and_window(self):
        fig, ax = plot_single(
            self.data, 0, dx=1.0, dt=0.01,
            start_time=0.0, end_time=0.5, norm_ref=2.0, show=False,
        )

        y = ax.lines[0].get_ydata()
        self.assertEqual(y.size, 51)
        expected = (self.data[0, :51] - self.data[0, :51].mean()) / 2.0
        np.testing.assert_allclose(y, expected)
        self.assertEqual(ax.get_ylabel(), "Amplitude / raw std")

    def test_draws_into_existing_axes(self):
        fig, ax = plt.subplots()
        out_fig, out_ax = plot_single(self.data, 1, dx=1.0, dt=0.01, ax=ax, show=False)
        self.assertIs(out_ax, ax)
        self.assertIs(out_fig, fig)


class OptionalDependencyTests(unittest.TestCase):
    def test_matplotlib_import_error_is_actionable(self):
        """Without matplotlib, calling a plot function must explain how to install it."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", fake_import):
            with self.assertRaisesRegex(ImportError, r"dasgauge\[plotting\]"):
                plotting._pyplot()


if __name__ == "__main__":
    unittest.main()
