# Luv_finder

Blind spectral-line finding in the UV plane for interferometric (ALMA) data.

A Gaussian line template is matched against the visibilities at every trial
position, source shape and line width, and the response is reported in
signal-to-noise units. A jackknifed copy of the data is the noise reference on the
same grid. Mosaics are corrected for the primary beam and combined, and candidates
are catalogued and fitted in the visibilities. Works on CASA measurement sets or on
NPZ exports.

Documentation: https://joshiwavm.github.io/Luv_finder/

## Install

```bash
conda create -n luv python=3.12 && conda activate luv
git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
pip install -e ".[casa,dev]"
```

CASA needs a runtime data directory; see the
[installation page](https://joshiwavm.github.io/Luv_finder/installation.html).

## Quickstart

The `smoke` preset uses the ACA and a 24-channel cube and runs in seconds:

```bash
luv-mock configs/mocks/smoke.yaml
luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms \
         --grid configs/grids/smoke.yaml --jackknife
```

The peak height of the response is the line's S/N, and the jackknife trace shows
the noise on the same axis. Mosaics, the catalogue and the line fit are in the
documentation.

## Development

```bash
pre-commit install && pre-commit run --all-files   # ruff lint + format
pytest                                             # CASA-free, uses a bundled fixture
pytest -m casa                                     # simulates the smoke preset (needs CASA)
pytest --plots                                     # also write plots/ + index.html to eyeball
make -C docs html                                  # docs -> docs/build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute, [ROADMAP.md](ROADMAP.md) for
planned work, and [CITATION.cff](CITATION.cff) to cite the software.
