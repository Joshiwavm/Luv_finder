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

Start with the smoke preset. It uses the 10-antenna ACA and a 16-channel cube,
so it runs in about 10 seconds and produces ~7 MB instead of ~180 MB:

```bash
luv-mock configs/mocks/smoke.yaml
luv-find --ms output/ms_files/smoke/smoke.aca.cycle10.noisy.ms \
         --grid configs/grids/smoke.yaml --jackknife
```

The science-scale presets use the 43-antenna 12 m array:

```bash
luv-mock configs/mocks/line13_line9.yaml
luv-export --ms output/ms_files/line13_line9/line13_line9.alma.cycle10.3.noisy.ms \
           --out line13_line9.npz
luv-find --ms line13_line9.npz --grid configs/grids/known_sources.yaml --jackknife
```

`configs/README.md` explains the presets, how mocks pair with grids, and the
sign flip between the image and model conventions for `dra`.

## Development

```bash
pre-commit install && pre-commit run --all-files   # ruff lint + format
pytest                                             # CASA-free tests on a bundled fixture
pytest -m casa                                     # regenerate a tiny mock (needs CASA)
make -C docs html                                  # docs -> docs/build/html
```

See [ROADMAP.md](ROADMAP.md) for planned work.
