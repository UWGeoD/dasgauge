"""
dasgauge — DAS (Distributed Acoustic Sensing) reading, preprocessing and plotting.

Portions of this package are derived from UWGeoD/DAS_Preprocessing at source
commit 896e00005b68c7878ae80d50833bd9eeabe7ebc7 and have been reorganized for
dasgauge. See PROVENANCE.md for the exact source files, line ranges, symbols,
modifications and licensing. The two repositories are maintained independently
after that import.
"""

from .io import DAS
from .preprocessing import (
    Step,
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
from .plotting import plot_das_data, plot_single

__version__ = "0.1.0"

__all__ = [
    "DAS",
    "Step",
    "bandpass_sos",
    "curvelet_denoise",
    "detrend_linear",
    "detrend_then_bandpass",
    "downsample_time",
    "fk_filter",
    "hilbert_transform",
    "integrate_strain_rate_fd",
    "integrate_to_strain",
    "make_preprocess",
    "plot_das_data",
    "plot_single",
    "__version__",
]
