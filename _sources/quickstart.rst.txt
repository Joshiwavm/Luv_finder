Quickstart
==========

A quick look
------------

``luv-find`` searches a measurement set or NPZ, corrects every pointing for its
primary beam, combines them, saves the best grid point's S/N spectrum to ``plots/``
and prints it. The ``smoke`` preset (ACA, 24 channels) runs in seconds:

.. code-block:: bash

   luv-mock configs/mocks/smoke.yaml
   luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms --grid configs/grids/smoke.yaml --jackknife

``--field N`` searches one pointing, ``--out response.npz`` keeps the responses, and
``--continuum-order`` and ``--cores`` set the filter.

A catalogue
-----------

Catalogues come from a run: one YAML file in, every product in one directory.

.. code-block:: yaml

   # run.yaml in the repository root; relative paths are relative to this file
   data: output/ms_files/line13_line9/line13_line9.alma.cycle13.3.noisy.ms
   output: output/line13_line9
   grid: configs/grids/line13_line9.yaml   # or the grid itself, as a mapping

Optional keys are ``name``, ``pb_limit``, ``max_channels`` (the channels a unit of the
search holds at most), ``filter`` (``continuum_order``, ``channel_noise``),
``reliability`` (``floor``, ``k``, ``fidelity_min``, ``rule``), ``near_miss``
(``fidelity``, ``max``) and ``cores``; an unknown key raises an error.

.. code-block:: bash

   luv-run run.yaml search      # export, search and catalogue every unit; --workers N shares them
   luv-run run.yaml fit         # fit the detections and near misses in the visibilities
   luv-run run.yaml summary
   luv-run run.yaml plots

The output holds ``catalogue.ecsv`` (every group, with its likelihood ratio, fidelity
and ``detected``), ``lines_fit.ecsv`` (the fitted lines), ``line_maps.npz`` (their
moment-0 maps), ``export/`` and ``units/`` (each unit's catalogue, maps and weights).
What is on disk is not redone, and ``Run.open(output)`` reopens a run for its figures.

Python
------

.. code-block:: python

   from luv_finder.data import load
   from luv_finder.matchedfilter import MatchedFilter, build_grid

   data = load("line13_line9.npz", [0])
   mf = MatchedFilter(data, build_grid(data, {"width": [200.0, 300.0]}))
   mf.run(jackknife=True)
   print(mf.best_params, mf.response.max())

``mf.response`` is the statistic T(x): unit variance under H0, so the peak height of a
line is its SNR, and ``mf.response_jackknife`` shows the noise on the same axis.
Column ``i`` is a template centred on trial frequency ``i``; rows follow
``mf.grid_params``, the product of the ``dra, ddec, bmin, bmaj, pa, width`` axes in
that order. A mock's declared ``snr`` is the SNR the filter recovers.
