"""
DAS plotting utilities.

Selected functionality adapted from UWGeoD/DAS_Preprocessing,
source commit 896e00005b68c7878ae80d50833bd9eeabe7ebc7, original file
`Utilities.py` (module constants at lines 9-10, ``plot_das_data`` at lines
201-249, ``plot_single`` at lines 252-296).

The code may have been reorganized for dasgauge. See PROVENANCE.md for exact
source ranges, imported symbols, licensing, and modifications.

Original work Copyright (c) 2025 GeoD Research Group, MIT License.

matplotlib is an optional dependency of dasgauge (extra: ``plotting``). It is
imported lazily so that importing this module never fails; only calling a
plotting function without matplotlib installed raises a clear error.
"""

import numpy as np

__all__ = ["plot_das_data", "plot_single", "normalize"]

epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)


def _pyplot():
    """Return ``matplotlib.pyplot``, or raise a clear error if unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "dasgauge plotting requires matplotlib, which is an optional "
            "dependency. Install it with:\n"
            "    pip install 'dasgauge[plotting]'\n"
            "or:\n"
            "    pip install matplotlib"
        ) from exc
    return plt


def plot_das_data(data, channels, dx, dt,
                  start_time=None, end_time=None,
                  title=None,
                  ax=None, fig=None, show=True):
    """
    Plot a DAS record as a channel-position vs. time heatmap.

    If ax is provided, draw into that axes.
    Otherwise create a new fig, ax.

    Returns (fig, ax).
    """
    plt = _pyplot()

    x = np.asarray(channels) * dx
    t = np.arange(data.shape[1]) * dt

    if start_time is not None and end_time is not None:
        mask = (t >= start_time) & (t <= end_time)
        data = data[:, mask]
        t = t[mask]

    # Create fig/ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    elif fig is None:
        fig = ax.figure

    ax.imshow(
        normalize(data).T,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[x[0], x[-1], t[-1], t[0]],
        interpolation="none",
        animated=True,
        zorder=1,
    )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t[-1], t[0])
    ax.set_xlabel("Channel Position (m)")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0.7)
    if show:
        plt.show()

    return fig, ax


def plot_single(
    data, channel_num, dx, dt,
    start_time=None, end_time=None,
    norm_ref=None,
    ax=None, show=True
):
    """
    Plot a single DAS channel waveform.

    Parameters
    ----------
    norm_ref : float or None
        If provided, the trace is centred (mean subtracted) then divided by norm_ref.
        Pass the std of the raw channel so that all rows in a comparison share the same
        amplitude reference and the raw row sits at ±1 by definition.
        If None, the raw amplitude is plotted as-is.
    """
    plt = _pyplot()

    t = np.arange(data.shape[1]) * dt
    sig = data[channel_num].copy().astype(float)

    if start_time is not None and end_time is not None:
        m = (t >= start_time) & (t <= end_time)
        t = t[m]
        sig = sig[m]

    if norm_ref is not None:
        sig = (sig - sig.mean()) / norm_ref

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(t, sig, color="black")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_title(f"Channel {channel_num} (Position: {channel_num*dx:.2f} m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude" if norm_ref is None else "Amplitude / raw std")
    ax.grid(True)

    if show:
        fig.tight_layout()
        plt.show()

    return fig, ax
