Quickstart
==========

Command line
------------

The ``smoke`` preset is deliberately tiny (ACA, 24 channels) and runs end to end
in seconds:

.. code-block:: bash

   luv-mock configs/mocks/smoke.yaml
   luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms --grid configs/grids/smoke.yaml --jackknife

The science preset uses the 12 m array. Exporting the measurement set to NPZ once
lets every later step run without CASA:

.. code-block:: bash

   luv-mock configs/mocks/line13_line9.yaml
   luv-export --ms output/ms_files/line13_line9/line13_line9.alma.cycle13.3.noisy.ms --out line13_line9.npz
   luv-find --ms line13_line9.npz --grid configs/grids/line13_line9.yaml --jackknife --out response.npz

``luv-find`` prints the best grid point and saves its S/N spectrum to ``plots/``.

Python
------

.. code-block:: python

   from luv_finder.data import load
   from luv_finder.matchedfilter import MatchedFilter, build_grid, grid_model

   data = load("line13_line9.npz")
   grid = build_grid(data, {"width": [200.0, 300.0]})
   mf = MatchedFilter(data, grid_model(grid))
   mf.run(jackknife=True)
   print(mf.best_params, mf.response.max())

Reading the output
------------------

``mf.response`` is the matched-filter statistic: unit variance under the null
hypothesis, so the peak height of a line is its signal-to-noise ratio, and
``mf.response_jackknife`` shows the noise on the same axis. Column ``i`` is a
line template centred on channel ``i``; rows follow ``mf.grid_params``, the
product of the ``dra, ddec, bmin, bmaj, pa, width`` axes in that order. A mock's
declared ``snr`` is the S/N the filter recovers.

For a mosaic, the catalogue and the line fit, see the guides.
