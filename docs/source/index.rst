Luv_finder
==========

Blind spectral-line search in the uv plane for interferometric (ALMA) data.

At every trial position, source shape and line width, Luv_finder matches a
Gaussian line template against the visibilities and reports the matched-filter
statistic in signal-to-noise units. A jackknife of the data, consecutive
integrations differenced, is the noise reference: it calibrates the S/N, the
shape a source leaves in the search, and the false-alarm rate. Mosaics are
corrected for the primary beam and combined, and candidates are catalogued with
a likelihood ratio and a fidelity, then fitted in the visibilities.

.. code-block:: bash

   pip install -e ".[casa]"
   luv-mock configs/mocks/smoke.yaml
   luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms --grid configs/grids/smoke.yaml --jackknife
   luv-run run.yaml search && luv-run run.yaml fit   # a catalogue, see the quickstart

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart
   method

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/mosaics
   guides/catalogue
   guides/configuration

.. toctree::
   :maxdepth: 1
   :caption: Reference

   tutorials
   api

Source: https://github.com/Joshiwavm/Luv_finder
