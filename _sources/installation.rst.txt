Installation
============

.. code-block:: bash

   conda create -n luv python=3.12 && conda activate luv
   git clone https://github.com/Joshiwavm/Luv_finder && cd Luv_finder
   pip install -e ".[casa,dev]"

Extras
------

``casa``
   ``casatools``, ``casatasks``, ``casadata``. Needed to read measurement sets and
   to simulate mocks. Native arm64 wheels exist for Python 3.12 on macOS >= 14.
   ``casadata`` downloads ~350 MB on first import; if it cannot write to
   ``~/.casa/data``, create that directory by hand. If the import fails with
   ``Symbol not found: _CRYPTO_calloc``, upgrade the environment's OpenSSL:
   ``conda install -c conda-forge "openssl>=3.6"`` (casatools bundles OpenSSL 3.6
   and dyld reuses whichever ``libcrypto.3.dylib`` loads first).
``jax``
   ``jax``, ``jax-finufft``. Not used yet; see the roadmap.
``dev``
   pytest, ruff, pre-commit, Sphinx.

The science code downstream of :mod:`luv_finder.data` is numpy-only. Exporting a
measurement set to NPZ with ``luv-export`` lets everything else run without CASA.
