Installation
============

.. code-block:: bash

   conda create -n luv python=3.12 && conda activate luv
   git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
   pip install -e ".[casa,dev]"

Extras
------

``casa``
   ``casatools`` and ``casatasks``, needed to read measurement sets and to simulate
   mocks. Native arm64 wheels exist for Python 3.12 on macOS >= 14.
``dev``
   pytest, ruff, pre-commit, Sphinx.

Reading data and simulating mocks are NumPy (and CASA); the matched filter is JAX, in
float64. A run exports its measurement set to NPZ once, so every later step runs without
CASA.

CASA setup
----------

CASA needs a runtime data directory (measures tables, ephemerides, antenna
configurations). Keep one shared copy instead of a ``casadata`` package per environment,
and point every environment at it from ``~/.casa/config.py``:

.. code-block:: python

   measurespath = "/Users/<you>/.casa/data"
   measures_auto_update = True   # small, changes often
   data_auto_update = False      # ~850 MB, update by hand

Populate it once with ``python -c "from casaconfig import pull_data; pull_data()"``.
Cycle 13 antenna configurations need casarundata 2026.02.19 or newer.

- If ``import casatools`` fails with ``Symbol not found: _CRYPTO_calloc``, upgrade the
  environment's OpenSSL: ``conda install -c conda-forge "openssl>=3.6"``.
- If ``casadata`` is installed as well, it shadows ``measurespath``: ``pip uninstall casadata``.
- CASA's ``casa-<timestamp>.log`` files go to ``logs/`` (gitignored; set ``LUV_CASA_LOG_DIR``
  to change it), as long as CASA is reached through ``luv_finder._casa``.
