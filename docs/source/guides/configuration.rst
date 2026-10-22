Configuration
=============

Search grids
------------

A grid YAML (``luv-find --grid``, a run's ``grid``, or the ``cfg`` of
:func:`~luv_finder.matchedfilter.build_grid`) sets the search axes. A value is a scalar (fixed), a list, or ``{start, stop, step}``;
the grid is the product of all axes, so the cost is too. Omitted keys follow from the
data:

``dra``, ``ddec``
   arcsec east and north of the reference; default a lattice in steps of half the
   resolution out to where the primary beam falls to ``pb_limit``.
``pb_limit``
   primary-beam level that bounds a pointing's search (default 0.2).
``bmaj``, ``bmin``, ``pa``
   the trial source's axes (FWHM, arcsec) and position angle (degrees east of north);
   default 0.2355 of the resolution (a sigma of a tenth), effectively a point source.
``width``
   line FWHM in km/s; default 100, 200, 300 and 400.

The line centre is not an axis: every template is centred on every channel. Any other
key raises an error; ``total_flux`` in particular cancels in the normalisation.
:class:`~luv_finder.matchedfilter.MatchedFilter` takes the grid as a dict of the same
keys, a value or a list each; a key left out takes its value from ``GRID_DEFAULTS``.

.. code-block:: yaml

   # configs/grids/default.yaml (comments trimmed): a blind search
   pb_limit: 0.2
   width: [100, 200, 300, 400] # km/s

Filter options
--------------

``continuum_order`` (``luv-find --continuum-order``)
   degree of the polynomial continuum fitted with each line template per window and
   position (default 1; ``none`` turns it off). A line loses the part of its S/N the
   polynomial can mimic, a few per cent. Lines at S/N >= 5 are left out of the continuum
   of the others at their position.
``channel_noise``
   ``"measure"`` (default) measures the channel correlation on the jackknife per spectral
   window; ``None`` assumes independent channels; or pass ``{spw: ChannelNoise}``.
``cores`` (``luv-find --cores``)
   the search pins itself to a quarter of the cores, at most all but two, because the
   machines are shared (Linux only).

Mock presets
------------

``configs/mocks/`` holds :class:`~luv_finder.mock.MockObservation` inputs
(``luv-mock <file>``) and ``configs/grids/`` the matching search grids, paired by file
name. A mock's declared ``snr`` is the optimal S/N, calibrated against the simulated
data. Antenna configurations are not vendored: ``alma_config`` names a file CASA ships,
such as ``alma.cycle13.3.cfg``. The presets and their pairing are listed in
`configs/README.md <https://github.com/Joshiwavm/Luv_finder/blob/main/configs/README.md>`_.

Runs
----

A run YAML (:doc:`../quickstart`) holds ``data`` and ``output`` and optionally ``grid``,
``pb_limit``, ``max_channels``, ``filter``, ``reliability``, ``near_miss`` and ``cores``.
The search is planned in units: windows that overlap or whose gap is narrower than the
widest template are one unit, so a line across a window edge is searched and fitted
whole. ``max_channels`` cuts a long unit into chunks of whole blocks (windows overlapping
by more than a channel) that load their neighbours and catalogue only the groups peaking
in their own part, so the chunks give the unchopped catalogue.
