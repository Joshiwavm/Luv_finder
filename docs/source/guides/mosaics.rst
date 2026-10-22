Mosaics and the primary beam
============================

Data layout
-----------

:class:`~luv_finder.data.DataHandler` holds one :class:`~luv_finder.data.Chunk` per
(field, spectral window): the weighted visibilities ``X = w * V`` and the weight of every
visibility ``w`` (0 where flagged) as ``(n_chan, n_row)`` arrays, ``u``, ``v``, time,
baseline and scan per row and frequency per channel. The weights are Stokes I of
``WEIGHT_SPECTRUM`` (or ``WEIGHT``), corrected on reading per scan and channel by the
noise measured from pairs of each baseline's integrations. Autocorrelations and flagged
rows are dropped.

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

Each pointing covers the region where its primary beam is at least ``pb_limit``
(``utils.PB_LIMIT``, 0.2). A dataset with a single field is primary-beam corrected too, so everything
downstream sees PB-corrected results. The steps are available separately as
:func:`~luv_finder.matchedfilter.build_grid`,
:meth:`MatchedFilter.run <luv_finder.matchedfilter.MatchedFilter.run>`,
:func:`~luv_finder.matchedfilter.pb_corrected` and
:func:`~luv_finder.matchedfilter.combine_pointings`.

From the command line:

.. code-block:: bash

   luv-find --ms mosaic.ms --grid grid.yaml --jackknife --out mosaic_response.npz
   luv-find --ms mosaic.ms --field 1 --grid grid.yaml --out field1.npz

``luv-find`` goes through ``search_pointings``: every field, or the one ``--field``
names, PB-corrected and combined, and reports the best grid point. A run does the same
for its catalogue.

The primary beam
----------------

:func:`luv_finder.utils.primary_beam` is an Airy pattern scaled to the FWHM the ALMA
Technical Handbook gives for the real antennas, 1.13 lambda/D, its series near the
pointing centre so the fit's derivatives hold there. Dividing by it leaves a
pointing's S/N unchanged; the combination weights every pointing by PB^2/sigma^2, so
overlaps gain S/N, and a pointing contributes nothing where its PB is below
``pb_limit``.

Dirty cubes and maps
--------------------

For inspection, :mod:`luv_finder.imaging` makes natural-weighted dirty cubes on the
search grid by type-1 NUFFT (:func:`~luv_finder.imaging.dirty_cube`), moment-8 and
continuum maps (:func:`~luv_finder.imaging.dirty_maps`) and their linear mosaic
(:func:`~luv_finder.imaging.mosaic_dirty_maps`). The search does not use them.
