# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Blind spectral-line finding in the UV plane for ALMA data: a UV-domain Gaussian line
model is grid-searched against visibilities with a matched filter; a jackknifed copy of
the data gives the noise reference. Only simulated single-pointing data so far.

## Environment and commands

- `conda activate luv` (arm64 Python 3.12, native; no Rosetta). Install: `pip install -e ".[casa,dev]"`.
- Linux (Allegro): `luv` lives at `/almastorage/allegro6/marrewijk/miniconda3/envs/luv`; activate by full path from bash, as in the parent `ALMA_analysis/CLAUDE.md`. casatools/casatasks are pinned to 6.7.5.18 in `pyproject.toml`: the 6.7.6.14 wheel ships no libcrypto and `import casatasks` fails. Do not work around it with `LD_PRELOAD`; it breaks Python's `ssl`.
- Run everything from the repo root; configs and outputs use root-relative paths.
- `pytest` — CASA-free, uses `tests/fixtures/line13_line9_small.npz`. `pytest -m casa` simulates the smoke preset (needs CASA). `pytest --plots` also writes figures and a contact sheet to `plots/`.
- `pre-commit run --all-files` — ruff lint + format, nbstripout. Keep it clean.
- `make -C docs html` — Sphinx (furo, nbsphinx); tutorials in `docs/source/tutorials/` are committed executed and never re-run by the build.
- CLIs: `luv-mock <yaml>`, `luv-export --ms X --out X.npz`, `luv-find --ms X[.npz] [--grid yaml] [--jackknife]`.

## Architecture

- `luv_finder/data.py` — `DataHandler`: flattens an MS into channel-major arrays (`uvdata`), NPZ round-trip, phase shift, `jackknife(mode="scan"|"random")`. CASA is reached lazily through `_casa.py`, so the NPZ path works without it.
- `luv_finder/model.py` — `Model` + `Gaussian` (2D spatial x 1D spectral, evaluated analytically in UV). Parameters are exposed as `src_{NN}_{attr}`; the matched filter routes values back via `key.split("_", 2)[-1]`.
- `luv_finder/matchedfilter.py` — `MatchedFilter`: expands `Model.grid` (scalar = fixed, array = enumerate, callable = derived from width), multiprocessing over grid points, FFT delay transform. The response is in S/N units (unit variance under the null); `total_flux` cancels in the kernel normalisation and is rejected as a grid key. `weighting="natural"` (noise weights only; the trial size cancels) or `"template"` (also weights by the source envelope A(u,v), optimal for resolved sources). Mock `snr` is the optimal S/N.
- `luv_finder/mock.py` — `MockObservation`: cube -> `simobserve` -> `tclean` -> `output/ms_files/<name>/`. YAML presets in `configs/mocks/`.
- `luv_finder/_casa.py` — the only place that imports `casatools`/`casatasks`. It presets `casaconfig.config.logfile` so CASA writes to `logs/` (override with `LUV_CASA_LOG_DIR`) instead of scattering `casa-<timestamp>.log` in the working directory. Never import CASA directly; use `tools()`/`tasks()`, or call `configure_logging()` first if you must.
- `luv_finder/plotting.py` — all diagnostic figures; each takes a directory and returns the path written. `MatchedFilter.plot_response` is a thin wrapper.
- `luv_finder/cli/` — argparse wrappers only; no science logic.
- `configs/mocks/` and `configs/grids/` pair by filename; `configs/README.md` covers the pairing and the `dra` sign flip. Antenna configs are not vendored: presets name files CASA ships (`alma.cycle13.3.cfg`), resolved by `mock.resolve_antenna_config`. `data/ output/ support/ plots/ logs/` are gitignored products.
- Test configs live in `configs/`, not under `tests/`: `tests/test_mock.py` loads `configs/mocks/smoke.yaml` and `configs/grids/smoke.yaml` rather than hardcoding parameters, so presets and tests cannot drift apart.

## Conventions

- DRY and KISS: one implementation per concept, thin wrappers, no speculative abstractions.
- Everything downstream of `data.py` is numpy-only so a JAX port stays mechanical.
- No nested sampling or other samplers; grid search only until ROADMAP.md says otherwise.
- No notebooks outside `docs/source/tutorials/`. New workflows become CLI commands + tests.
- Real-data organisation/concatenation belongs to `alma-data-prep`, not here.
- Planned work and open problems (JAX, catalog, primary beam + mosaics): see ROADMAP.md.
- Notion write-ups of the method (the Luv-finder page) read like a JOSS paper but are not for submission: explanation, math and code excerpts interleaved. Use the terminology of Vio & Andreani (2021, arXiv:2107.09378): H0/H1, template s, statistic T(x), SNR, P_FA, threshold gamma. Concise and accurate; state each thing once.
