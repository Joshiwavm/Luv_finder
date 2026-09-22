Usage
=====

Command line
------------

.. code-block:: bash

   luv-mock configs/mocks/line13_line9.yaml
   luv-export --ms output/ms_files/line13_line9/line13_line9.alma.cycle10.3.noisy.ms --out line13_line9.npz
   luv-find --ms line13_line9.npz --grid configs/grids/default.yaml --jackknife --out response.npz

Python
------

.. code-block:: python

   import functools
   from luv_finder import DataHandler, Gaussian, MatchedFilter, Model
   from luv_finder.matchedfilter import nu_center_func

   data = DataHandler.from_npz("line13_line9.npz")
   res = data.metadata.minresolution()

   g = Gaussian()
   g.grid = {
       "dra": [4.25, 12.0], "ddec": [23.5, -4.25],
       "bmin": res / 10, "bmaj": res / 10,
       "width": [300.0], "total_flux": [1.0],
       "nu_center": functools.partial(nu_center_func, uvfreq_min=data.uvdata.uvfreqs.min()),
   }
   mod = Model(); mod.addcomponent(g)

   mf = MatchedFilter(data, mod)
   mf.run(jackknife=True)
   mf.plot_response("filter_response.png")
   print(mf.best_params)

Parameter naming
----------------

Every component attribute is exposed as ``src_{index:02d}_{name}``
(``src_00_dra``, ``src_00_nu_center``, ...). Grid dictionaries on a component are
prefixed the same way when added to a :class:`~luv_finder.model.Model`. A grid
entry may be a scalar (fixed), an array (enumerated), or a callable of the width
(derived, e.g. :func:`~luv_finder.matchedfilter.nu_center_func`).
