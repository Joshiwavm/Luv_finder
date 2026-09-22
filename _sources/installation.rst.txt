Installation
============

.. code-block:: bash

   conda create -n luv python=3.12 && conda activate luv
   git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
   pip install -e ".[casa,dev]"

Extras
------

``casa``
   ``casatools`` and ``casatasks``. Needed to read measurement sets and to
   simulate mocks. Native arm64 wheels exist for Python 3.12 on macOS >= 14.
   If the import fails with ``Symbol not found: _CRYPTO_calloc``, upgrade the
   environment's OpenSSL: ``conda install -c conda-forge "openssl>=3.6"``
   (casatools bundles OpenSSL 3.6 and dyld reuses whichever
   ``libcrypto.3.dylib`` loads first).
``jax``
   ``jax``, ``jax-finufft``. Not used yet; see the roadmap.
``dev``
   pytest, ruff, pre-commit, Sphinx.

The science code downstream of :mod:`luv_finder.data` is numpy-only. Exporting a
measurement set to NPZ with ``luv-export`` lets everything else run without CASA.

CASA runtime data
-----------------

CASA needs a data directory (measures tables, ephemerides, observatory
positions). The ``casadata`` pip package ships one copy per environment at
roughly 850 MB each; it is not a dependency of ``casatools`` and this project
does not install it. Keep a single shared copy instead, maintained by
casaconfig, and point every environment at it from ``~/.casa/config.py``:

.. code-block:: python

   measurespath = "/Users/<you>/.casa/data"
   measures_auto_update = True   # small, changes often
   data_auto_update = False      # ~850 MB, update by hand

Antenna configurations come from this directory too, so the repository does not
vendor any ``.cfg`` files. Cycle 13 requires casarundata 2026.02.19 or newer;
``luv_finder.mock.resolve_antenna_config`` reports which cycles your copy has if
the requested one is missing.

Populate or refresh it with:

.. code-block:: bash

   python -c "from casaconfig import pull_data; pull_data()"       # first time
   python -c "from casaconfig import measures_update; measures_update()"
   python -c "from casaconfig import data_update; data_update()"

If ``casadata`` is installed alongside this, it shadows ``measurespath`` on the
data path and CASA warns that the two sets of measures tables differ. Uninstall
it: ``pip uninstall casadata``.

CASA log files
--------------

``casatools`` and ``casatasks`` write a ``casa-<timestamp>.log`` into the working
directory as soon as they are imported. :mod:`luv_finder._casa` presets the
casaconfig log path so these land in ``logs/`` instead, which is gitignored. Set
``LUV_CASA_LOG_DIR`` to send them elsewhere. Reach CASA through
``luv_finder._casa.tools()`` / ``tasks()``; importing it directly re-creates the
clutter, because the destination is fixed at the first CASA import.
