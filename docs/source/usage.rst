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
(``src_00_dra``, ``src_00_nu_center``, ...). Grid dictionaries on a component are
prefixed the same way when added to a :class:`~luv_finder.model.Model`. A grid
entry may be a scalar (fixed), an array (enumerated), or a callable of the width
(derived, e.g. :func:`~luv_finder.matchedfilter.nu_center_func`).
