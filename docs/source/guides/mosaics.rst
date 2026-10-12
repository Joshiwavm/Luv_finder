Mosaics and the primary beam
============================

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

Searching a mosaic
------------------

A mosaic is searched one pointing at a time, on one sky grid.
:func:`~luv_finder.matchedfilter.search_pointings` loads each field of an NPZ or
measurement set in turn, searches it, corrects it for its primary beam and combines the
pointings:

.. code-block:: python

   from luv_finder.matchedfilter import search_pointings

   result = search_pointings("mosaic.npz", {"width": [200.0, 300.0]}, jackknife=True)

Each pointing covers the region where its primary beam is at least ``pb_limit`` (0.2 by
default). A dataset with a single field is primary-beam corrected too, so everything
downstream sees PB-corrected results. The steps are available separately as
:func:`~luv_finder.matchedfilter.build_grid`,
:meth:`MatchedFilter.run <luv_finder.matchedfilter.MatchedFilter.run>`,
:func:`~luv_finder.matchedfilter.pb_corrected` and
:func:`~luv_finder.matchedfilter.combine_pointings`.

From the command line:

.. code-block:: bash

   luv-export --ms mosaic.ms --out mosaic.npz
   luv-find --ms mosaic.npz --grid grid.yaml --jackknife --out mosaic_response.npz
   luv-find --ms mosaic.npz --field 1 --grid grid.yaml --out field1.npz

Without ``--field`` a dataset with several fields goes through ``search_pointings``, and
the best grid point of the combined, PB-corrected result is reported. With ``--field``,
or for a single field, one pointing is searched as it is; it is PB-corrected only when a
``--catalogue`` is written.

The primary beam
----------------

:func:`luv_finder.utils.primary_beam` is an Airy pattern scaled to the FWHM the ALMA
Technical Handbook gives for the real antennas, 1.13 lambda/D. Dividing by it leaves a
pointing's S/N unchanged; the combination weights every pointing by PB^2/sigma^2, so
overlaps gain S/N, and a pointing contributes nothing where its PB is below
``pb_limit``.

Dirty cubes and maps
--------------------

For inspection, :mod:`luv_finder.imaging` makes natural-weighted dirty cubes on the
search grid by type-1 NUFFT (:func:`~luv_finder.imaging.dirty_cube`), moment-8 and
continuum maps (:func:`~luv_finder.imaging.dirty_maps`) and their linear mosaic
(:func:`~luv_finder.imaging.mosaic_dirty_maps`). The search does not use them.
