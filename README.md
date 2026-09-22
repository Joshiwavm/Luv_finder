# Luv_finder

Blind spectral-line finding in the UV plane for interferometric (ALMA) data.

A parametric line model (2D spatial x 1D spectral Gaussian) is evaluated directly on
the visibilities and cross-correlated with the data over a grid of positions, sizes,
widths and fluxes. A jackknifed copy of the data gives a signal-free noise
realisation for the same grid. Works on CASA measurement sets or on NPZ exports.

Documentation: https://joshiwavm.github.io/Luv_finder/

## Install

```bash
conda create -n luv python=3.12 && conda activate luv
git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
pip install -e ".[casa,dev]"        # add ,jax for the JAX extras
```

Runs natively on Apple Silicon (casatools ships arm64 wheels for Python 3.12,
macOS >= 14).

CASA needs a runtime data directory. Keep one shared copy rather than a
per-environment `casadata` package, by setting `measurespath` in `~/.casa/config.py`:

```python
measurespath = "/Users/<you>/.casa/data"
measures_auto_update = True
data_auto_update = False
```

Populate it once with `python -c "from casaconfig import pull_data; pull_data()"`.

If `import casatools` fails with `Symbol not found: _CRYPTO_calloc`, the conda env's
OpenSSL is older than the one casatools bundles: `conda install -c conda-forge "openssl>=3.6"`.

## Quickstart

The `smoke` preset uses the ACA and a 24-channel cube: ~9 MB and ~20 s, against
~180 MB for the science preset.

```bash
luv-mock configs/mocks/smoke.yaml
luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms \
         --grid configs/grids/smoke.yaml --jackknife
```

The science preset uses the 43-antenna 12 m array:

```bash
luv-mock configs/mocks/line13_line9.yaml
luv-export --ms output/ms_files/line13_line9/line13_line9.alma.cycle13.3.noisy.ms \
           --out line13_line9.npz
luv-find --ms line13_line9.npz --grid configs/grids/line13_line9.yaml --jackknife
```

The filter response is in signal-to-noise units, so the peak height is the line's
S/N and the jackknife trace shows the noise floor on the same axis. A mock's
declared `snr` is calibrated against the simulated data, so it means the S/N you
actually get. See [configs/README.md](configs/README.md).

Antenna configurations are not vendored: `alma_config` names a file CASA ships,
such as `alma.cycle13.3.cfg`. Cycle 13 needs casarundata 2026.02.19 or newer.

## Development

```bash
pre-commit install && pre-commit run --all-files   # ruff lint + format
pytest                                             # CASA-free, uses a bundled fixture
pytest -m casa                                     # simulates the smoke preset (needs CASA)
pytest --plots                                     # also write plots/ + index.html to eyeball
make -C docs html                                  # docs -> docs/build/html
```

See [ROADMAP.md](ROADMAP.md) for planned work.
