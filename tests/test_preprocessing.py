"""Tests for dasgauge.preprocessing on small synthetic arrays.

test_downsample_updates_fs_for_later_steps is adapted from
UWGeoD/DAS_Preprocessing tests/test_preprocessing.py at source commit
896e00005b68c7878ae80d50833bd9eeabe7ebc7 (imports rewritten for dasgauge).
"""

import unittest

import numpy as np

from dasgauge.preprocessing import (
    bandpass_sos,
    curvelet_denoise,
    detrend_linear,
    detrend_then_bandpass,
    downsample_time,
    fk_filter,
    hilbert_transform,
    integrate_strain_rate_fd,
    integrate_to_strain,
    make_preprocess,
)


def _sine(n_ch=4, n_t=512, fs=100.0, freq=5.0):
    t = np.arange(n_t) / fs
    return np.tile(np.sin(2 * np.pi * freq * t), (n_ch, 1)).astype(np.float64)


class DetrendTests(unittest.TestCase):
    def test_detrend_removes_linear_ramp(self):
        n_t = 200
        ramp = np.linspace(0.0, 10.0, n_t)
        x = np.vstack([ramp, ramp * -2.0 + 3.0])

        y = detrend_linear(x)

        np.testing.assert_allclose(y, 0.0, atol=1e-9)

    def test_detrend_rejects_non_2d(self):
        with self.assertRaisesRegex(ValueError, "Expected 2D array"):
            detrend_linear(np.zeros(10))


class BandpassTests(unittest.TestCase):
    def test_bandpass_attenuates_out_of_band_tone(self):
        fs = 200.0
        in_band = _sine(n_ch=2, n_t=2048, fs=fs, freq=10.0)
        out_band = _sine(n_ch=2, n_t=2048, fs=fs, freq=80.0)

        kw = dict(fs=fs, f_lo=5.0, f_hi=20.0, order=5, zero_phase=True)
        kept = bandpass_sos(in_band, **kw)
        killed = bandpass_sos(out_band, **kw)

        # ignore filter edge transients
        self.assertGreater(kept[:, 200:-200].std(), 0.5)
        self.assertLess(killed[:, 200:-200].std(), 0.01)

    def test_bandpass_validates_band(self):
        with self.assertRaisesRegex(ValueError, r"0 < f_lo < f_hi < fs/2"):
            bandpass_sos(_sine(), fs=100.0, f_lo=30.0, f_hi=10.0)
        with self.assertRaisesRegex(ValueError, "order must be"):
            bandpass_sos(_sine(), fs=100.0, f_lo=1.0, f_hi=10.0, order=0)

    def test_detrend_then_bandpass_matches_composition(self):
        fs = 100.0
        x = _sine(fs=fs) + np.linspace(0, 5, 512)

        combined = detrend_then_bandpass(x, fs=fs, f_lo=1.0, f_hi=20.0)
        manual = bandpass_sos(detrend_linear(x), fs=fs, f_lo=1.0, f_hi=20.0)

        np.testing.assert_allclose(combined, manual)


class FkFilterTests(unittest.TestCase):
    def test_returns_real_same_shape_and_preserves_mean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(32, 128)) + 5.0

        y = fk_filter(x, dx=1.0, dt=0.01)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isrealobj(y))
        # DC bin is explicitly preserved, so the overall mean survives
        self.assertAlmostEqual(y.mean(), x.mean(), places=6)

    @staticmethod
    def _moveout_wavelet(n_ch=64, n_t=512, dx=5.0, dt=0.002, f0=25.0, v=1000.0):
        """Band-limited Gaussian-tapered wavelet with linear moveout at `v` m/s.

        Energy concentrates near f=f0, k=f0/v, i.e. apparent velocity v.
        """
        t = np.arange(n_t) * dt
        x = np.zeros((n_ch, n_t))
        for ch in range(n_ch):
            tc = 0.3 + ch * dx / v
            env = np.exp(-((t - tc) ** 2) / (2 * 0.03 ** 2))
            x[ch] = env * np.sin(2 * np.pi * f0 * (t - tc))
        return x

    def test_velocity_band_keeps_matching_moveout(self):
        x = self._moveout_wavelet(v=1000.0)
        e_in = np.sum(x ** 2)

        y = fk_filter(x, dx=5.0, dt=0.002, vmin=700.0, vmax=1500.0)

        # a 1000 m/s event survives a 700-1500 m/s pass band nearly intact
        self.assertGreater(np.sum(y ** 2) / e_in, 0.9)

    def test_velocity_band_rejects_non_matching_moveout(self):
        x = self._moveout_wavelet(v=1000.0)
        e_in = np.sum(x ** 2)

        fast = fk_filter(x, dx=5.0, dt=0.002, vmin=3000.0, vmax=6000.0)
        slow = fk_filter(x, dx=5.0, dt=0.002, vmin=100.0, vmax=400.0)

        self.assertLess(np.sum(fast ** 2) / e_in, 0.05)
        self.assertLess(np.sum(slow ** 2) / e_in, 0.05)

    def test_frequency_band_rejects_out_of_band_tone(self):
        n_ch, n_t = 64, 512
        dx, dt = 5.0, 0.002
        t = np.arange(n_t) * dt
        in_band = np.tile(np.sin(2 * np.pi * 5.0 * t), (n_ch, 1))
        out_band = np.tile(np.sin(2 * np.pi * 40.0 * t), (n_ch, 1))

        kept = fk_filter(in_band, dx=dx, dt=dt, fmin=1.0, fmax=10.0)
        killed = fk_filter(out_band, dx=dx, dt=dt, fmin=1.0, fmax=10.0)

        self.assertGreater(np.sum(kept ** 2) / np.sum(in_band ** 2), 0.9)
        self.assertLess(np.sum(killed ** 2) / np.sum(out_band ** 2), 1e-3)


class IntegrationTests(unittest.TestCase):
    def test_integrate_to_strain_of_cosine_is_sine(self):
        fs = 500.0
        n_t = 2000
        t = np.arange(n_t) / fs
        f0 = 4.0
        x = np.cos(2 * np.pi * f0 * t)[None, :]

        y = integrate_to_strain(x, fs=fs)
        expected = np.sin(2 * np.pi * f0 * t) / (2 * np.pi * f0)

        np.testing.assert_allclose(y[0], expected, atol=2e-3)

    def test_integrate_fd_matches_analytic_up_to_sign_convention(self):
        fs = 500.0
        n_t = 2000
        t = np.arange(n_t) / fs
        f0 = 5.0  # integer number of periods -> no spectral leakage
        x = np.cos(2 * np.pi * f0 * t)[None, :]

        y = integrate_strain_rate_fd(x, fs=fs)
        # author's convention carries a leading minus sign
        expected = -np.sin(2 * np.pi * f0 * t) / (2 * np.pi * f0)

        np.testing.assert_allclose(y[0], expected, atol=1e-8)

    def test_integrate_fd_zeroes_dc(self):
        x = np.ones((2, 128))
        y = integrate_strain_rate_fd(x, fs=100.0)
        np.testing.assert_allclose(y, 0.0, atol=1e-12)


class DownsampleTests(unittest.TestCase):
    def test_decimates_by_integer_factor(self):
        x = np.arange(20, dtype=np.float64).reshape(2, 10)

        y = downsample_time(x, fs=100.0, target_fs=50.0)

        self.assertEqual(y.shape, (2, 5))
        np.testing.assert_array_equal(y, x[:, ::2])

    def test_noop_when_target_ge_fs(self):
        x = np.arange(20, dtype=np.float64).reshape(2, 10)
        np.testing.assert_array_equal(downsample_time(x, fs=50.0, target_fs=50.0), x)


class HilbertTests(unittest.TestCase):
    def test_envelope_of_am_signal(self):
        fs = 1000.0
        t = np.arange(4096) / fs
        env = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
        x = (env * np.sin(2 * np.pi * 100.0 * t))[None, :]

        y = hilbert_transform(x, mode="envelope")

        self.assertTrue(np.all(y >= 0))
        np.testing.assert_allclose(y[0, 500:-500], env[500:-500], atol=0.05)

    def test_modes_and_validation(self):
        x = _sine()
        self.assertLessEqual(hilbert_transform(x, mode="phase").max(), np.pi)
        np.testing.assert_allclose(hilbert_transform(x, mode="real"), x, atol=1e-9)
        with self.assertRaisesRegex(ValueError, "Unknown hilbert mode"):
            hilbert_transform(x, mode="nope")


class CurveletTests(unittest.TestCase):
    def test_runs_without_external_library(self):
        """curvelet_denoise is pure NumPy — no CurveLab/curvelops needed."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(32, 32))

        y = curvelet_denoise(x, n_scales=3, thresh=1.0)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isrealobj(y))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_zero_threshold_approximately_reconstructs(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(32, 32))

        y = curvelet_denoise(x, n_scales=3, thresh=0.0, absolute=True, keep_lowpass=True)

        # tight-frame synthesis with no thresholding should be close to identity
        self.assertGreater(np.corrcoef(x.ravel(), y.ravel())[0, 1], 0.95)

    def test_reduces_energy_of_pure_noise(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(32, 32))

        y = curvelet_denoise(x, n_scales=3, thresh=3.0, robust_mad=True)

        self.assertLess(np.sum(y ** 2), np.sum(x ** 2))


class PipelineTests(unittest.TestCase):
    def test_empty_pipeline_is_identity(self):
        x = _sine()
        np.testing.assert_allclose(make_preprocess()(x, 100.0), x)

    def test_runs_full_chain(self):
        fs = 500.0
        rng = np.random.default_rng(4)
        x = rng.normal(size=(16, 1024))

        pp = make_preprocess([
            ("detrend", {}),
            ("bandpass", {"f_lo": 1.0, "f_hi": 20.0, "order": 4}),
            ("fk_filter", {"vmin": 100.0, "vmax": 5000.0}),
            ("hilbert", {"mode": "real"}),
            ("downsample", {"target_fs": 50.0}),
        ], dx=1.0)

        y = pp(x, fs)

        # decimation is x[:, ::10] over 1024 samples -> ceil(1024/10) = 103
        self.assertEqual(y.shape, (16, len(range(0, 1024, 10))))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_dict_spec_matches_list_spec(self):
        fs = 200.0
        x = _sine(fs=fs)
        steps_kw = {"f_lo": 1.0, "f_hi": 20.0}

        from_list = make_preprocess([("detrend", {}), ("bandpass", steps_kw)])(x, fs)
        from_dict = make_preprocess({"detrend": {}, "bandpass": steps_kw})(x, fs)

        np.testing.assert_allclose(from_list, from_dict)

    def test_rejects_unknown_step(self):
        with self.assertRaisesRegex(ValueError, "Unknown step"):
            make_preprocess([("nope", {})])

    def test_fk_filter_step_requires_dx(self):
        pp = make_preprocess([("fk_filter", {})])
        with self.assertRaisesRegex(ValueError, "fk_filter requires dx"):
            pp(_sine(), 100.0)
        # supplying dx at call time works
        self.assertEqual(pp(_sine(), 100.0, dx=1.0).shape, (4, 512))

    def test_downsample_updates_fs_for_later_steps(self):
        """Adapted from DAS_Preprocessing tests/test_preprocessing.py."""
        fs = 5000.0
        target_fs = 50.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 5000)).astype(np.float32)

        pp = make_preprocess([
            ("downsample", {"target_fs": target_fs}),
            ("integrate_fd", {"tukey_alpha": 0.0}),
        ])

        actual = pp(x, fs)
        expected = integrate_strain_rate_fd(x[:, ::100], fs=target_fs, tukey_alpha=0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
