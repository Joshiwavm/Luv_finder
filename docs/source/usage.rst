Usage
=====

Command line
------------

The ``smoke`` preset is deliberately tiny (ACA, 24 channels, one minute) and
runs end to end in seconds:

.. code-block:: bash

   luv-mock configs/mocks/smoke.yaml
   luv-find --ms output/ms_files/smoke/smoke.aca.cycle13.noisy.ms --grid configs/grids/smoke.yaml --jackknife

The science preset uses the 12 m array and 50 channels:

.. code-block:: bash

   luv-mock configs/mocks/line13_line9.yaml
   luv-export --ms output/ms_files/line13_line9/line13_line9.alma.cycle13.3.noisy.ms --out line13_line9.npz
   luv-find --ms line13_line9.npz --grid configs/grids/line13_line9.yaml --jackknife --out response.npz

Each mock preset has a grid preset of the same name. ``configs/README.md``
documents the pairing and the ``dra`` sign flip.

A mosaic is searched one pointing at a time, on one sky grid:

.. code-block:: bash

   luv-export --ms mosaic.ms --out mosaic.npz
   luv-find --ms mosaic.npz --field 0 --grid grid.yaml --out field0.npz
   luv-find --ms mosaic.npz --field 1 --grid grid.yaml --out field1.npz

Data layout
-----------

:class:`~luv_finder.data.DataHandler` holds one :class:`~luv_finder.data.Chunk` per
(field, spectral window): the weighted visibilities ``X = w * V`` and their flags as
``(n_chan, n_row)`` arrays, and ``u``, ``v``, time and weight once per row and frequency
once per channel. The weight is the measurement set's ``WEIGHT`` column (Stokes I of the
parallel hands); flags zero it per channel. Autocorrelations and flagged rows are dropped
on reading.

Positions (``dra``, ``ddec``) are arcsec east and north of one reference direction per
dataset, the phase centre of its first target field. Every chunk knows its own field's
phase centre in that frame, so a grid point is the same sky position in every pointing,
and the default position grid is a lattice anchored at the reference.

Signal-to-noise units
---------------------

``MatchedFilter.response`` is the matched-filter statistic: unit variance under
the null, so the peak height is the line's signal-to-noise ratio. It does not
depend on the template amplitude, which cancels in the kernel normalisation, so
``total_flux`` is not a search axis and passing it in a grid raises an error.

Diagnostic figures
------------------

.. code-block:: bash

   pytest --plots

writes the visibility-spectrum and filter-response checks to ``plots/`` together
with an ``index.html`` contact sheet. The same figures are available directly
through :func:`luv_finder.plotting.spectrum_check` and
:func:`luv_finder.plotting.response_check`.

Parameter naming
----------------

Every component attribute is exposed as ``src_{index:02d}_{name}``
(``src_00_dra``, ``src_00_width``, ...). Grid dictionaries on a component are
prefixed the same way when added to a :class:`~luv_finder.model.Model`. A grid
entry may be a scalar (fixed) or an array (enumerated). The line centre is not a
grid axis: the filter places its kernel in every spectral window and slides it
across the window.
