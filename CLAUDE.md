# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Blind spectral-line finding in the UV plane for ALMA data: a UV-domain Gaussian line
model is grid-searched against visibilities with a matched filter; a jackknifed copy of
the data gives the noise reference. Only simulated single-pointing data so far.

## Environment and commands

- `conda activate luv` (arm64 Python 3.12, native; no Rosetta). Install: `pip install -e ".[casa,dev]"`.
- Run everything from the repo root; configs and outputs use root-relative paths.
- `pytest` — CASA-free, uses `tests/fixtures/line13_line9_small.npz`. `pytest -m casa` regenerates a tiny mock (needs CASA).
- `pre-commit run --all-files` — ruff lint + format, nbstripout. Keep it clean.
- `make -C docs html` — Sphinx (furo, nbsphinx); tutorials in `docs/source/tutorials/` are committed executed and never re-run by the build.
- CLIs: `luv-mock <yaml>`, `luv-export --ms X --out X.npz`, `luv-find --ms X[.npz] [--grid yaml] [--jackknife]`.

## Architecture

- `luv_finder/data.py` — `DataHandler`: flattens an MS into channel-major arrays (`uvdata`), NPZ round-trip, phase shift, `jackknife(mode="scan"|"random")`. CASA is reached lazily through `_casa.py`, so the NPZ path works without it.
- `luv_finder/model.py` — `Model` + `Gaussian` (2D spatial x 1D spectral, evaluated analytically in UV). Parameters are exposed as `src_{NN}_{attr}`; the matched filter routes values back via `key.split("_", 2)[-1]`.
- `luv_finder/matchedfilter.py` — `MatchedFilter`: expands `Model.grid` (scalar = fixed, array = enumerate, callable = derived from width), multiprocessing over grid points, FFT delay transform. `RESPONSE_SCALE = 0.9` is an unexplained empirical factor.
- `luv_finder/mock.py` — `MockObservation`: cube -> `simobserve` -> `tclean` -> `output/ms_files/<name>/`. YAML presets in `configs/mocks/`.
- `luv_finder/_casa.py` — the only place that imports `casatools`/`casatasks`. It presets `casaconfig.config.logfile` so CASA writes to `logs/` (override with `LUV_CASA_LOG_DIR`) instead of scattering `casa-<timestamp>.log` in the working directory. Never import CASA directly; use `tools()`/`tasks()`, or call `configure_logging()` first if you must.
- `luv_finder/cli/` — argparse wrappers only; no science logic.
- `configs/alma/` antenna configs; `configs/grids/` grid presets; `data/ output/ support/ plots/` are gitignored products.

## Conventions

- DRY and KISS: one implementation per concept, thin wrappers, no speculative abstractions.
- Everything downstream of `data.py` is numpy-only so a JAX port stays mechanical.
- No nested sampling or other samplers; grid search only until ROADMAP.md says otherwise.
- No notebooks outside `docs/source/tutorials/`. New workflows become CLI commands + tests.
- Real-data organisation/concatenation belongs to `alma-data-prep`, not here.
- Planned work and open problems (JAX, catalog, primary beam + mosaics): see ROADMAP.md.
