# Contributing

Bug reports, questions and pull requests are welcome on the
[issue tracker](https://github.com/Joshiwavm/Luv_finder/issues). For a bug, include the
command or code you ran and, if you can, a small NPZ export (`luv-export`) that shows it.

## Setup

```bash
conda create -n luv python=3.12 && conda activate luv
git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
pip install -e ".[casa,dev]"
pre-commit install
```

## Before opening a pull request

```bash
pre-commit run --all-files   # ruff lint + format, nbstripout
pytest                       # CASA-free, on the bundled fixture
pytest -m casa               # if you touched the mocks or the CASA I/O
make -C docs html            # if you touched the documentation or docstrings
```

Add a line under `[Unreleased]` in `CHANGELOG.md`.

## Conventions

- One implementation per concept: thin wrappers, no speculative abstractions. The
  command-line tools in `luv_finder/cli/` hold no science logic.
- CASA is imported only through `luv_finder/_casa.py`, so everything that reads an NPZ
  runs without it.
- Data I/O, the model and the mocks are NumPy/SciPy; the filter, the imaging and the fit's
  derivatives are JAX. Formulas used by both (`model.envelope`, `utils.primary_beam`) take
  an `xp=` argument instead of being written twice.
- A new workflow becomes a command-line option and a test, not a notebook. Tutorials in
  `docs/source/tutorials/` are committed executed.
- Test presets live in `configs/`, paired by file name between `mocks/` and `grids/`.

Planned work and open problems are in [ROADMAP.md](ROADMAP.md). The maintainer is
Joshiwa van Marrewijk; questions go through the issue tracker.
