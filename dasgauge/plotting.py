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

__all__ = ["plot_das_data", "plot_single", "plot_waterfall", "normalize"]

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
                  ax=None, fig=None, show=True,
                  channel_axis="distance"):
    """
    Plot a DAS record as a channel vs. time heatmap.

    channel_axis selects what the horizontal axis measures:
    "distance" (default) scales `channels` by dx and labels metres;
    "number" plots the raw channel indices instead.

    If ax is provided, draw into that axes.
    Otherwise create a new fig, ax.

    Returns (fig, ax).
    """
    plt = _pyplot()

    if channel_axis == "distance":
        x = np.asarray(channels) * dx
        xlabel = "Channel Position (m)"
    elif channel_axis == "number":
        x = np.asarray(channels, dtype=float)
        xlabel = "Channel Number"
    else:
        raise ValueError(
            f"channel_axis must be 'distance' or 'number', got {channel_axis!r}"
        )
    x_step = float(x[1] - x[0]) if x.size > 1 else (dx if channel_axis == "distance" else 1.0)
    t = np.arange(data.shape[1]) * dt

    if start_time is not None and end_time is not None:
        mask = (t >= start_time) & (t <= end_time)
        data = data[:, mask]
        t = t[mask]

    # Create fig/ax if not provided.  When callers supply axes (for example,
    # a multi-panel manuscript figure), leave global layout management to
    # them rather than applying tight_layout after each individual panel.
    created_axes = ax is None
    if created_axes:
        fig, ax = plt.subplots(figsize=(10, 5))
    elif fig is None:
        fig = ax.figure

    ax.imshow(
        normalize(data).T,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[x[0] - x_step / 2, x[-1] + x_step / 2, t[-1] + dt / 2, t[0] - dt / 2],
        interpolation="none",
        animated=True,
        zorder=1,
    )
    ax.set_xlim(x[0] - x_step / 2, x[-1] + x_step / 2)
    ax.set_ylim(t[-1] + dt / 2, t[0] - dt / 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)

    if created_axes:
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


def plot_waterfall(
    data,
    channels,
    dt,
    start_time=None,
    end_time=None,
    gain=None,
    demean=True,
    color="black",
    linewidth=0.45,
    title=None,
    ax=None,
    fig=None,
    show=True,
):
    """Plot one waveform per DAS channel, vertically offset by channel index.

    A coherent moveout appears as an aligned feature crossing the offset
    traces. Pass the same explicit ``gain`` to multiple panels when their
    relative amplitudes must be visually comparable. If ``gain`` is omitted,
    it is chosen from the displayed panel's 99th absolute percentile.

    Parameters
    ----------
    data : array-like, shape (n_channels, n_time)
        DAS channel-by-time array.
    channels : array-like, shape (n_channels,)
        Absolute channel indices used as vertical trace offsets.
    dt : float
        Sampling interval in seconds.
    gain : float or None
        Vertical channel-index units per data-amplitude unit. The plotted
        ordinate is ``channel - gain * trace``.
    demean : bool
        Subtract each displayed channel's temporal mean before plotting.
    """
    plt = _pyplot()
    values = np.asarray(data)
    channel_indices = np.asarray(channels, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D [channel, time] array")
    if channel_indices.ndim != 1 or len(channel_indices) != values.shape[0]:
        raise ValueError("channels must be one-dimensional and match data rows")
    if values.shape[1] == 0:
        raise ValueError("data must contain at least one time sample")
    if float(dt) <= 0:
        raise ValueError("dt must be positive")

    time = np.arange(values.shape[1], dtype=float) * float(dt)
    mask = np.ones(values.shape[1], dtype=bool)
    if start_time is not None:
        mask &= time >= float(start_time)
    if end_time is not None:
        mask &= time <= float(end_time)
    if not np.any(mask):
        raise ValueError("requested time window contains no samples")
    time = time[mask]
    traces = values[:, mask].astype(float, copy=True)
    if demean:
        traces -= traces.mean(axis=1, keepdims=True)

    if gain is None:
        reference = float(np.quantile(np.abs(traces), 0.99))
        gain = 1.0 if reference <= epsilon else 0.9 / reference
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")

    created_axes = ax is None
    if created_axes:
        fig, ax = plt.subplots(figsize=(10, 5))
    elif fig is None:
        fig = ax.figure

    for row, channel in enumerate(channel_indices):
        ax.plot(
            time,
            channel - gain * traces[row],
            color=color,
            linewidth=linewidth,
        )
    margin = 0.5 if len(channel_indices) == 1 else 0.5 * np.median(np.diff(channel_indices))
    ax.set_ylim(channel_indices[-1] + margin, channel_indices[0] - margin)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index (traces offset by channel)")
    if title:
        ax.set_title(title)

    if created_axes:
        fig.tight_layout(pad=0.7)
    if show:
        plt.show()
    return fig, ax
