"""
DAS preprocessing utilities.

Selected functionality adapted from UWGeoD/DAS_Preprocessing,
source commit 896e00005b68c7878ae80d50833bd9eeabe7ebc7, original file
`preprocessing.py` (lines 38-52, 58-345, 348-508, 519-542, 545-553, 556-667).

The code may have been reorganized for dasgauge. See PROVENANCE.md for exact
source ranges, imported symbols, licensing, and modifications.

Original work Copyright (c) 2025 GeoD Research Group, MIT License.

Design goals
------------
- Clear, composable preprocessing "pipeline" API (like sklearn.Pipeline, but lightweight).
- Each step is a pure function: x -> x (2D arrays: [channels, time]).
- A single builder `make_preprocess(...)` returns a callable:
      preprocess(x, fs, **ctx) -> y
  so you can plug it directly into your Dataset / notebook workflows.

Conventions
-----------
- Input data shape: (n_channels, n_time)
- Default time axis is axis=1
- fs in Hz, dt = 1/fs (unless provided)
- dx in meters per channel (only needed for f-k filter)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid


Array2D = np.ndarray

__all__ = [
    "detrend_linear",
    "bandpass_sos",
    "detrend_then_bandpass",
    "fk_filter",
    "integrate_to_strain",
    "integrate_strain_rate_fd",
    "downsample_time",
    "hilbert_transform",
    "curvelet_denoise",
    "Step",
    "make_preprocess",
]


# =============================================================================
# Validation helpers
# =============================================================================
def _as_2d_float(x: np.ndarray) -> Array2D:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [channels, time], got shape={x.shape}")
    # keep dtype unless it's not float-ish
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)
    return x


def _require_positive(name: str, val: float) -> float:
    val = float(val)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}")
    return val


# =============================================================================
# Core processing primitives (stateless functions)
# =============================================================================
def detrend_linear(x: Array2D, *, axis: int = 1) -> Array2D:
    """Linear detrend along `axis` (default time axis=1)."""
    x = _as_2d_float(x)
    return signal.detrend(x, type="linear", axis=axis)


def bandpass_sos(
    x: Array2D,
    *,
    fs: float,
    f_lo: float = 1.0,
    f_hi: float = 20.0,
    order: int = 5,
    axis: int = 1,
    zero_phase: bool = False,
) -> Array2D:
    """
    Butterworth bandpass via SOS.

    Parameters
    ----------
    zero_phase:
        If True, uses sosfiltfilt (zero-phase, more expensive).
        If False, uses sosfilt (causal, faster).
    """
    x = _as_2d_float(x)
    fs = _require_positive("fs", fs)
    f_lo = float(f_lo)
    f_hi = float(f_hi)
    if not (0 < f_lo < f_hi < fs / 2):
        raise ValueError(f"Require 0 < f_lo < f_hi < fs/2. Got f_lo={f_lo}, f_hi={f_hi}, fs={fs}")

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    sos = signal.butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")

    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=axis)
    return signal.sosfilt(sos, x, axis=axis)


def detrend_then_bandpass(
    x: Array2D,
    *,
    fs: float,
    f_lo: float = 1.0,
    f_hi: float = 20.0,
    order: int = 5,
    axis: int = 1,
    zero_phase: bool = False,
) -> Array2D:
    """Convenience: detrend -> bandpass."""
    return bandpass_sos(
        detrend_linear(x, axis=axis),
        fs=fs,
        f_lo=f_lo,
        f_hi=f_hi,
        order=order,
        axis=axis,
        zero_phase=zero_phase,
    )


def fk_filter(
    x: Array2D,
    *,
    dx: float,
    dt: float,
    fmin: float = 0.0,
    fmax: float = None,
    vmin: float = None,
    vmax: float = None,
    taper: float = 0.0,
) -> Array2D:
    """
    Refined 2D f-k filter.

    Tapering strategy: "Inner Taper" (tapers INSIDE the defined bounds).
    """
    x = np.asarray(x, dtype=float)
    nch, nt = x.shape

    # 1. Use RFFT2 (Real FFT) - faster and saves memory for real input
    # Note: RFFT2 only computes positive frequencies for the time axis
    X = np.fft.rfft2(x)

    # Shift only the spatial axis (k), time axis (f) starts at 0 naturally in RFFT
    X = np.fft.fftshift(X, axes=0)

    # 2. Define Frequency Grids
    # kx: centered from -k_nyquist to +k_nyquist
    kx = np.fft.fftshift(np.fft.fftfreq(nch, d=dx))
    # f: positive frequencies only (0 to f_nyquist)
    f = np.fft.rfftfreq(nt, d=dt)

    KX, F = np.meshgrid(kx, f, indexing="ij")
    absK = np.abs(KX)
    absF = np.abs(F)

    # Avoid division by zero
    eps = 1e-12
    vapp = absF / np.maximum(absK, eps)

    # 3. Create Mask
    M = np.ones_like(X, dtype=float)

    # Helper for inner tapering
    def apply_taper(mask, values, low, high, width):
        if width <= 0:
            if low is not None: mask *= (values >= low)
            if high is not None: mask *= (values <= high)
            return mask

        # Taper Logic
        if low is not None:
            # Ramp from 0 at low to 1 at low*(1+w)
            # We must kill everything below low
            mask[values < low] = 0.0
            ramp_up = low * (1 + width)
            idx = (values >= low) & (values < ramp_up)
            # Hanning-like ramp
            mask[idx] *= 0.5 * (1 - np.cos(np.pi * (values[idx] - low) / (ramp_up - low)))

        if high is not None:
            # Ramp from 1 at high*(1-w) to 0 at high
            mask[values > high] = 0.0
            ramp_down = high * (1 - width)
            idx = (values <= high) & (values > ramp_down)
            mask[idx] *= 0.5 * (1 + np.cos(np.pi * (values[idx] - ramp_down) / (high - ramp_down)))

        return mask

    # Apply Filters
    M = apply_taper(M, absF, fmin, fmax, taper)

    # Velocity filter requires care with vmin/vmax
    # If vmin is set, we want to REMOVE slow stuff (v < vmin)
    # If vmax is set, we want to REMOVE fast stuff (v > vmax)
    M = apply_taper(M, vapp, vmin, vmax, taper)

    # 4. Handle DC cleanly
    # We always preserve true static DC (f=0, k=0) because it's the mean.
    # We DO NOT force the whole k=0 axis to 1 unless naturally included.
    # Note: RFFT means f=0 is at index 0 on axis 1.
    # X is shifted on axis 0, so k=0 is at index nch//2.

    # Ensure mean is preserved (optional, but usually good)
    center_k = nch // 2
    M[center_k, 0] = 1.0

    # 5. Apply and Inverse
    Y = X * M

    # Unshift k-axis before inverse
    Y = np.fft.ifftshift(Y, axes=0)
    y = np.fft.irfft2(Y, s=x.shape) # s=x.shape ensures odd/even count is correct

    return y


def integrate_to_strain(
    x: Array2D,
    *,
    fs: float,
    axis: int = 1,
) -> Array2D:
    """
    Convert strain rate → strain by cumulative trapezoidal integration along the time axis.

    Per-channel demeaning is applied to the strain rate before integrating. Without this,
    any DC offset in the strain rate (the fiber's resting static pre-strain level) becomes
    a linear ramp in the integrated strain, dominating the signal amplitude and masking
    vehicle-induced perturbations. Demeaning first keeps the integration result centered
    near zero. Residual low-frequency drift is removed by the subsequent detrend/bandpass
    steps in the preprocessing pipeline.
    """
    x = _as_2d_float(x)
    dt = 1.0 / _require_positive("fs", fs)
    x = x - x.mean(axis=axis, keepdims=True)
    return cumulative_trapezoid(x, dx=dt, axis=axis, initial=0).astype(x.dtype)


def integrate_strain_rate_fd(
    x: Array2D,
    *,
    fs: float,
    axis: int = 1,
    tukey_alpha: float = 0.0,
) -> Array2D:
    """
    Convert strain rate → strain by frequency-domain integration.

    Matches van den Ende et al. (2023) Fig10 notebook:
      Y_int = -FFT(win * Y) / (2j * pi * freqs)
    The negative sign is the author's physics convention for this DAS dataset.
    Using +Y/(2j*pi*f) gives the opposite sign on all strain values, which
    reverses the gradient direction during downstream model training.

    Prefer this over cumulative-trapezoid integration (integrate_to_strain)
    for spectral fidelity: fd integration is exact for bandlimited signals
    and does not accumulate low-frequency drift over long windows.

    Use as a pipeline step ("integrate_fd") so it can be inserted at the
    appropriate position in the preprocessing chain, e.g. after bandpass and
    downsample (the order used in van den Ende 2023).
    """
    x = _as_2d_float(x)
    out_dtype = x.dtype
    fs = _require_positive("fs", fs)
    n_time = x.shape[axis]

    if tukey_alpha and tukey_alpha > 0:
        shape = [1] * x.ndim
        shape[axis] = n_time
        win = signal.windows.tukey(n_time, alpha=float(tukey_alpha)).astype(out_dtype, copy=False).reshape(shape)
        x = x * win

    X = np.fft.rfft(x, axis=axis)
    freqs = np.fft.rfftfreq(n_time, d=1.0 / fs)  # Hz

    shape = [1] * x.ndim
    shape[axis] = freqs.size
    freqs_bc = freqs.reshape(shape)

    # author's convention: Y_int = -Y / (2j*pi*f);  DC component → 0
    freqs_safe = np.where(freqs_bc == 0.0, 1.0, freqs_bc)   # replace DC with 1 to avoid division by zero
    X_int = -X / (2j * np.pi * freqs_safe)
    X_int = np.where(freqs_bc == 0.0, 0.0, X_int)           # zero DC explicitly

    return np.fft.irfft(X_int, n=n_time, axis=axis).astype(out_dtype)


def downsample_time(
    x: Array2D,
    *,
    fs: float,
    target_fs: float = 50.0,
) -> Array2D:
    """
    Decimate along the time axis by taking every n-th sample.

    Must be placed AFTER a bandpass step whose f_hi < target_fs / 2, so
    the bandpass acts as the anti-aliasing filter. No additional low-pass
    filter is applied here.

    Parameters
    ----------
    target_fs:
        Desired output rate (Hz). Actual rate = fs / round(fs / target_fs),
        which equals target_fs exactly when fs is an integer multiple of it.
    """
    x = _as_2d_float(x)
    factor = max(1, int(round(fs / target_fs)))
    if factor <= 1:
        return x
    return x[:, ::factor].astype(x.dtype)


def hilbert_transform(
    x: Array2D,
    *,
    axis: int = 1,
    mode: str = "envelope",
) -> Array2D:
    """
    Applies the Hilbert transform.

    Parameters
    ----------
    mode : {"envelope", "phase", "real"}
        "envelope" (default) returns the amplitude envelope of the signal.
        "phase" returns the instantaneous phase [-pi, pi].
        "real" returns the original signal.
    """
    x = _as_2d_float(x)
    # signal.hilbert computes the complex analytic signal
    analytic = signal.hilbert(x, axis=axis)

    if mode == "envelope":
        return np.abs(analytic).astype(x.dtype)
    elif mode == "phase":
        return np.angle(analytic).astype(x.dtype)
    elif mode == "real":
        return np.real(analytic).astype(x.dtype)
    else:
        raise ValueError(f"Unknown hilbert mode: '{mode}'. Use 'envelope' or 'phase'.")


def _meyer_nu(x: np.ndarray) -> np.ndarray:
    """C² smooth transition: ν(0)=0, ν(1)=1, ν(x)+ν(1-x)=1."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _meyer_W(r: np.ndarray) -> np.ndarray:
    """Radial Meyer window W(r), support [2/3, 5/3].

    Satisfies Σ_j W(2^{-j} r)² = 1 for r > 0 (partition of unity).
    From Candès et al. 2006, Sidebar 'Window Functions'.
    """
    r = np.abs(r)
    w = np.zeros_like(r, dtype=float)
    m1 = (r >= 2.0 / 3) & (r < 5.0 / 6)
    w[m1] = np.cos(np.pi / 2 * _meyer_nu(5.0 - 6.0 * r[m1]))
    w[(r >= 5.0 / 6) & (r <= 4.0 / 3)] = 1.0
    m2 = (r > 4.0 / 3) & (r <= 5.0 / 3)
    w[m2] = np.cos(np.pi / 2 * _meyer_nu(3.0 * r[m2] - 4.0))
    return w


def _meyer_V(omega: np.ndarray) -> np.ndarray:
    """Angular Meyer window V(ω), support [-2/3, 2/3].

    Satisfies Σ_l V(N·s + l)² = 1 for all s (periodization property).
    From Candès et al. 2006, Sidebar 'Window Functions'.
    """
    a = np.abs(omega)
    v = np.zeros_like(a, dtype=float)
    v[a <= 1.0 / 3] = 1.0
    m = (a > 1.0 / 3) & (a <= 2.0 / 3)
    v[m] = np.cos(np.pi / 2 * _meyer_nu(3.0 * a[m] - 1.0))
    return v


def _curvelet_soft_threshold(c: np.ndarray, thresh: float, robust_mad: bool, absolute: bool = False) -> np.ndarray:
    """Soft-threshold real spatial curvelet coefficients."""
    if absolute:
        lam = thresh
    elif robust_mad:
        nz = np.abs(c).ravel()
        nz = nz[nz > 0]
        sigma = float(np.median(nz)) / 0.6745 if nz.size > 0 else 0.0
        lam = thresh * sigma
    else:
        sigma = float(np.std(c))
        lam = thresh * sigma
    mag = np.abs(c)
    return np.sign(c) * np.maximum(0.0, mag - lam)


def curvelet_denoise(
    x: Array2D,
    *,
    n_scales: int = 4,
    thresh: float = 3.5,
    keep_lowpass: bool = True,
    robust_mad: bool = True,
    absolute: bool = False,
) -> Array2D:
    """
    Discrete curvelet-frame denoising following Candès et al. (2006).

    Implemented directly on top of NumPy's FFT — no external curvelet library
    (CurveLab / PyCurvelab / curvelops) is required.

    Key properties vs a naive directional FFT filter:
    - Parabolic scaling: N_j = 4·2^⌈j/2⌉ directional wedges at scale j
    - Meyer radial W(r) and angular V(ω) windows (partition of unity)
    - 4-cone shear geometry with equispaced slopes (not equispaced angles)
    - Centrosymmetric cone pairs (E+W, N+S) so IFFT gives real coefficients
    - Per-wedge spatial-domain soft-thresholding then tight-frame synthesis

    Radial normalization: scale_j = (10/3)·2^{n_scales-1-j}, so the finest
    scale (j=n_scales-1) covers |P| ∈ [0.2, 0.5] (up to Nyquist 0.5).

    Parameters
    ----------
    n_scales : int
        Number of dyadic scales. Default 4.
    thresh : float
        Threshold value. Interpretation depends on absolute/robust_mad:
          absolute=True  → lam = thresh directly (Zhang et al. 2025: thresh=0.3)
          absolute=False, robust_mad=True  → lam = thresh × MAD_sigma
          absolute=False, robust_mad=False → lam = thresh × std
    keep_lowpass : bool
        Pass frequencies below the coarsest scale through unmodified.
    robust_mad : bool
        Estimate σ via MAD (robust). Ignored when absolute=True.
    absolute : bool
        If True, use thresh as a direct coefficient threshold (no noise estimation).
    """
    x = _as_2d_float(x)
    ny, nx = x.shape
    n_scales = int(max(2, n_scales))

    F = np.fft.fftshift(np.fft.fft2(x))

    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    KY, FX = np.meshgrid(ky, fx, indexing="ij")

    F_out = np.zeros_like(F)
    # Accumulate radial coverage per primary axis to build the LP complement
    total_W2_FX = np.zeros((ny, nx), dtype=float)
    total_W2_KY = np.zeros((ny, nx), dtype=float)

    for j in range(n_scales):
        # Parabolic scaling: n_half = 2^{⌈j/2⌉} slopes per half-cone
        n_half = int(2 ** math.ceil(j / 2))

        # Radial scale: W(|P|*scale) has support [2/3, 5/3]/scale.
        # Finest scale (j=n_scales-1): scale=10/3 → |P| ∈ [0.2, 0.5] (Nyquist).
        scale = (10.0 / 3.0) * float(2 ** (n_scales - 1 - j))

        W_r_FX = _meyer_W(np.abs(FX) * scale)
        W_r_KY = _meyer_W(np.abs(KY) * scale)
        total_W2_FX += W_r_FX ** 2
        total_W2_KY += W_r_KY ** 2

        # Two centrosymmetric cone pairs for a real signal x:
        #   E+W pair (P=FX): E cone (FX>0, |FX|≥|KY|) paired with W cone (FX<0,
        #             mirrored slope -l) → centrosymmetric mask → IFFT is real.
        #   N+S pair (P=KY): strict |KY|>|FX| avoids diagonal double-count.
        for P, Q, W_r, use_strict in (
            (FX, KY, W_r_FX, False),
            (KY, FX, W_r_KY, True),
        ):
            abs_P = np.abs(P)
            safe_P = np.where(abs_P > 1e-12, abs_P, 1e-12)
            slope = Q / safe_P  # ξ_secondary / |ξ_primary|

            if use_strict:
                in_pos = (P > 0) & (abs_P > np.abs(Q))
                in_neg = (P < 0) & (abs_P > np.abs(Q))
            else:
                in_pos = (P > 0) & (abs_P >= np.abs(Q))
                in_neg = (P < 0) & (abs_P >= np.abs(Q))

            # Pair positive half-cone slope l with negative half-cone slope -l.
            # M(-ξ)=M(ξ) because V is even and cones are centrosymmetric → IFFT real.
            for l in range(-n_half, n_half + 1):
                V_pos = _meyer_V(float(n_half) * slope + float(l))
                V_neg = _meyer_V(float(n_half) * slope - float(l))
                M = W_r * (V_pos * in_pos.astype(float) + V_neg * in_neg.astype(float))
                if not np.any(M > 1e-8):
                    continue

                # Analysis → spatial threshold → synthesis (tight-frame)
                c = np.fft.ifft2(np.fft.ifftshift(F * M)).real
                c_thr = _curvelet_soft_threshold(c, thresh, robust_mad, absolute)
                F_out += np.fft.fftshift(np.fft.fft2(c_thr)) * M

    if keep_lowpass:
        # Synthesis adds F*M² per wedge, so total = F * Σ_j M_j².
        # For perfect reconstruction: Σ_j M_j² + lp_mask = 1
        # → lp_mask = 1 - Σ_j M_j² = 1 - total_W2 (NOT sqrt).
        in_E = (np.abs(FX) >= np.abs(KY))
        lp_mask = (np.maximum(0.0, 1.0 - total_W2_FX) * in_E.astype(float)
                   + np.maximum(0.0, 1.0 - total_W2_KY) * (~in_E).astype(float))
        F_out += F * lp_mask

    return np.fft.ifft2(np.fft.ifftshift(F_out)).real


# =============================================================================
# Pipeline API (recommended)
# =============================================================================
@dataclass(frozen=True)
class Step:
    """A single preprocessing step."""
    name: str
    fn: Callable[..., Array2D]
    kwargs: Dict[str, Any]


def _normalize_spec(
    spec: Optional[Union[Sequence[Tuple[str, Mapping[str, Any]]], Mapping[str, Mapping[str, Any]]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Accept either:
      - Ordered list: [("detrend", {...}), ("bandpass", {...}), ...]
      - Dict (py3.7+ preserves insertion order): {"detrend": {...}, "bandpass": {...}}
    Return a list of (name, kwargs) in execution order.
    """
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        return [(k, dict(v)) for k, v in spec.items()]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, kw in spec:
        out.append((name, dict(kw)))
    return out


_STEP_REGISTRY: Dict[str, Callable[..., Array2D]] = {
    "detrend": detrend_linear,
    "bandpass": bandpass_sos,
    "fk_filter": fk_filter,
    "curvelet": curvelet_denoise,
    "hilbert": hilbert_transform,
    "downsample": downsample_time,
    "integrate_fd": integrate_strain_rate_fd,
}


def make_preprocess(
    steps: Optional[
        Union[
            Sequence[Tuple[str, Mapping[str, Any]]],
            Mapping[str, Mapping[str, Any]],
        ]
    ] = None,
    *,
    axis: int = 1,
    dx: Optional[float] = None,
    dt: Optional[float] = None,
) -> Callable[[Array2D, float], Array2D]:
    """
    Build a preprocessing callable preprocess(x, fs, **ctx).

    Parameters
    ----------
    steps:
        Ordered pipeline specification. Recommended forms:

        1) List of tuples (explicit order):
           steps = [
             ("detrend", {"axis": 1}),
             ("bandpass", {"f_lo": 1, "f_hi": 20, "order": 5, "zero_phase": False}),
             ("fk_filter", {"fmin": 1, "fmax": 20, "vmin": 200, "vmax": 4000}),
             ("curvelet", {"n_scales": 4, "thresh": 3.5}),
           ]

        2) Dict (insertion order preserved in Python 3.7+):
           steps = {"detrend": {...}, "bandpass": {...}, ...}

    axis:
        Default axis for steps that need it (detrend/bandpass).

    dx, dt:
        Optional defaults captured in the closure (useful for fk_filter).
        - If dt is not provided, it will be computed from fs (dt = 1/fs).
        - fk_filter requires dx and dt to be available either here or at call time.

    Returns
    -------
    preprocess(x, fs, **ctx) -> y

    Notes
    -----
    - `bandpass` step automatically receives `fs`.
    - `fk_filter` step receives `dx` and `dt` (dt derived from fs unless provided).
    - You can override dx/dt at call time: preprocess(x, fs, dx=..., dt=...)
    """
    spec = _normalize_spec(steps)

    # Build normalized step list
    pipeline: List[Step] = []
    for name, kw in spec:
        if name not in _STEP_REGISTRY:
            raise ValueError(f"Unknown step '{name}'. Valid: {sorted(_STEP_REGISTRY)}")

        # common defaults
        if name in ("detrend", "bandpass", "hilbert", "integrate_fd"):
            kw.setdefault("axis", axis)
        # downsample has no axis kwarg — it always acts on axis=1

        # n_angles is no longer a parameter (replaced by parabolic scaling law)
        if name == "curvelet":
            kw.pop("n_angles", None)

        pipeline.append(Step(name=name, fn=_STEP_REGISTRY[name], kwargs=kw))

    def preprocess(x: Array2D, fs: float, **ctx: Any) -> Array2D:
        x2 = _as_2d_float(x)

        fs_ = _require_positive("fs", fs)
        current_fs = fs_
        dt_ = float(ctx.get("dt", dt if dt is not None else 1.0 / current_fs))
        dx_ = ctx.get("dx", dx)

        y = x2
        for step in pipeline:
            if step.name == "detrend":
                y = step.fn(y, **step.kwargs)

            elif step.name == "bandpass":
                y = step.fn(y, fs=current_fs, **step.kwargs)

            elif step.name == "fk_filter":
                if dx_ is None:
                    raise ValueError("fk_filter requires dx (meters). Provide dx in make_preprocess(dx=...) or at call time.")
                y = step.fn(y, dx=float(dx_), dt=float(dt_), **step.kwargs)

            elif step.name == "curvelet":
                y = step.fn(y, **step.kwargs)

            elif step.name == "hilbert":
                y = step.fn(y, **step.kwargs)

            elif step.name == "downsample":
                target_fs = float(step.kwargs.get("target_fs", 50.0))
                factor = max(1, int(round(current_fs / target_fs)))
                y = step.fn(y, fs=current_fs, **step.kwargs)
                current_fs = current_fs / factor
                dt_ = 1.0 / current_fs

            elif step.name == "integrate_fd":
                y = step.fn(y, fs=current_fs, **step.kwargs)

            else:
                # should never happen due to registry guard
                raise RuntimeError(f"Unhandled step: {step.name}")

        return y

    return preprocess
