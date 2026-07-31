# dasgauge

Reading, preprocessing and plotting utilities for Distributed Acoustic Sensing (DAS) data.

## Contents

| Module | Purpose |
| --- | --- |
| `dasgauge.io` | HDF5 readers for OptaSense, Silixa (DAS-RCN style), and generic/bare HDF5 files |
| `dasgauge.preprocessing` | Detrend, bandpass, f-k filter, downsample, integration, Hilbert transform, curvelet denoising, plus a composable pipeline builder |
| `dasgauge.plotting` | DAS heatmap and single-channel waveform plots |

## Install

```bash
pip install -e .              # core: numpy, scipy, h5py
pip install -e '.[plotting]'  # adds matplotlib for dasgauge.plotting
```

`matplotlib` is optional and imported lazily — `import dasgauge` works without
it, and only calling a plotting function raises a clear installation message.
Curvelet denoising is implemented on NumPy's FFT and needs no external
curvelet library.

## Quick start

```python
import numpy as np
from dasgauge import DAS, make_preprocess, plot_das_data

das = DAS("record.h5", vendor="OptaSense")

pp = make_preprocess([
    ("detrend", {}),
    ("bandpass", {"f_lo": 1.0, "f_hi": 15.0, "order": 5}),
    ("downsample", {"target_fs": 50.0}),
], dx=das.meta["dx"])

clean = pp(das.data, das.meta["fs"])
plot_das_data(clean, np.arange(clean.shape[0]), das.meta["dx"], 1 / 50.0, show=False)
```

## Tests

The suite is standard-library `unittest` — no extra test dependency:

```bash
python -m unittest discover -s tests -v
```

## Provenance

Parts of this package are derived from
[UWGeoD/DAS_Preprocessing](https://github.com/UWGeoD/DAS_Preprocessing) at commit
`896e00005b68c7878ae80d50833bd9eeabe7ebc7`. See [PROVENANCE.md](PROVENANCE.md) and
[provenance/DAS_Preprocessing.json](provenance/DAS_Preprocessing.json) for exact
source files, line ranges, symbols, modifications and licensing. The two
repositories are maintained independently after that import.

## License

MIT — see [LICENSE](LICENSE).
