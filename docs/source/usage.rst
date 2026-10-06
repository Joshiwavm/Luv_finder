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
documents the pairing.

A mosaic is searched one pointing at a time, on one sky grid. In Python
:func:`~luv_finder.matchedfilter.search_pointings` loads each field of an NPZ or measurement set
in turn, searches it, corrects it for its primary beam and combines the pointings:

.. code-block:: python

   from luv_finder.matchedfilter import search_pointings

   result = search_pointings("mosaic.npz", {"width": [200.0, 300.0]}, jackknife=True)

Each pointing covers the region where its primary beam is at least ``pb_limit``. A dataset with
a single field is primary-beam corrected too, so everything downstream sees PB-corrected
results. The steps are available separately as :func:`~luv_finder.matchedfilter.build_grid`,
:meth:`MatchedFilter.run <luv_finder.matchedfilter.MatchedFilter.run>`,
:func:`~luv_finder.matchedfilter.pb_corrected` and
:func:`~luv_finder.matchedfilter.combine_pointings`.

The primary beam (:func:`luv_finder.utils.primary_beam`) is an Airy pattern scaled to the
FWHM the ALMA Technical Handbook gives for the real antennas, 1.13 lambda/D. Dividing by it
leaves a pointing's S/N unchanged; the combination weights every pointing by PB^2/sigma^2,
so overlaps gain S/N, and a pointing contributes nothing where its PB is below ``pb_limit``.

From the command line:

.. code-block:: bash

   luv-export --ms mosaic.ms --out mosaic.npz
   luv-find --ms mosaic.npz --grid grid.yaml --jackknife --out mosaic_response.npz
   luv-find --ms mosaic.npz --field 1 --grid grid.yaml --out field1.npz
   luv-find --ms mosaic.npz --jackknife --catalogue lines.ecsv

Without ``--field`` a dataset with several fields goes through ``search_pointings``, and the
best grid point of the combined, PB-corrected result is reported. With ``--field`` (or a single
field) one pointing is searched as it is, without primary-beam correction.

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

Column ``i`` of the response is a line template centred exactly on channel ``i``.
Each column is normalised over the channels its template covers, so the variance
stays one at the window edges and next to flagged channels, and a line near an
edge is reported at the S/N of its observed part. Rows follow ``grid_params``:
the product of the ``dra, ddec, bmin, bmaj, pa, width`` axes, in that order.

At every trial position each line template is fitted jointly with a continuum: a polynomial in
frequency (degree 2 by default, ``continuum_order=`` or ``luv-find --continuum-order``; ``none``
turns it off) per spectral window, by weighted least squares on the dirty spectrum at that
position. A continuum source anywhere in the field is smooth in frequency at its own position,
however far it is from the phase centre. The joint fit is linear, keeps the response exactly
normalised and needs no line-free channels; a line loses the part of its S/N the polynomial
can mimic, a few per cent.

The search runs in JAX with float64, all positions of a window at once: on a
``dra x ddec`` grid the phase shift factorises, so each channel is one complex
matrix product, and the phase factors advance from channel to channel by
recurrence. Large grids are split into blocks of ``dra`` rows to bound memory.
On a shared machine the search pins itself to a quarter of the cores, never more
than all but two (``luv-find --cores`` or ``MatchedFilter.run(cores=...)`` to
change it; Linux only).

Catalogue
---------

:func:`luv_finder.catalogue.catalogue` turns a search with its jackknife (one pointing or the
combined mosaic, on a uniform position lattice) into an astropy table of line candidates; the
jackknife, which has the data's noise and no sky, is the reference throughout:

1. the S/N is the filter's. ALMA's Hanning spectral response correlates neighbouring channels
   (2/3 and 1/6); the search measures that correlation on the jackknife per spectral window
   (:func:`~luv_finder.data.channel_correlation`) and normalises every template by its exact
   variance, so the S/N has unit variance under the null
   (:func:`~luv_finder.catalogue.jackknife_spread` checks it). Data averaged in frequency, with
   less correlation, are measured the same way;
2. the correlation function of the jackknife response is measured; it is also the expected shape
   of a matched line's response (Vio & Andreani 2021), so its half-power ellipsoid in position and
   frequency is what one source occupies;
3. local maxima of the best-template S/N above 4 are merged, brightest first, within that
   ellipsoid, for the data, the jackknife and (as a diagnostic) the negated data;
4. every data group gets the likelihood ratio ``N_data(>= S/N) / N_jackknife(>= S/N)``
   (van Marrewijk et al. 2025, required >= 3) and the fidelity ``1 - N_jackknife / N_data`` per
   S/N bin, fitted with an error function (Walter et al. 2016, required >= 0.6).

.. code-block:: python

   from luv_finder.catalogue import catalogue

   cat = catalogue(result, ref=data.metadata.ref)
   lines = cat[cat["detected"]]
   cat.write("lines.ecsv", format="ascii.ecsv")

:func:`luv_finder.plotting.reliability_check` shows the counts, the likelihood ratio and the
fidelity; :func:`luv_finder.plotting.response_shape_check` the correlation against real lines.

Fitting the detected lines
--------------------------

Detection stays a grid search. :func:`luv_finder.fit.fit_lines` then refines every detected line
by least squares in the visibilities: an elliptical Gaussian source (position and sky covariance)
whose spectrum is a Gaussian line (peak, centre, FWHM) on a polynomial continuum of the search's
degree, times each pointing's primary beam at the source, jointly in every pointing whose primary
beam covers the line (PB >= 0.2), in the spectral window holding it.

Every derivative of the model in position and shape is the model times a polynomial in (u, v), so
one pass over the visibilities collects a few per-channel moments
(:func:`~luv_finder.kernel.point_moments`) that give chi^2, its gradient and the Gauss-Newton
matrix exactly. The spectrum (peak, continuum, centre, width) is fitted on those moments without
another pass, and ``scipy.optimize.minimize`` (L-BFGS-B) moves the position and shape on the
profiled chi^2, starting from the matched filter's peak: 4-16 passes per line. The shape is two
axis variances and an angle, bounded at zero (a flat prior on sizes >= 0), so an unresolved axis
lands on zero; its error, one-sided there, is NaN, and so is the angle of a point source.

.. code-block:: python

   from luv_finder.fit import fit_lines

   fitted = fit_lines("mosaic.npz", cat)  # or the search's DataHandler
   fitted.write("lines.ecsv", format="ascii.ecsv", overwrite=True)

or ``luv-find ... --jackknife --catalogue lines.ecsv --fit``. The fitted columns (``fit_dra``,
``fit_ddec``, ``fit_ra``, ``fit_dec``; ``fit_bmaj``, ``fit_bmin`` as sigma and ``fit_pa``;
``fit_freq_ghz``, ``fit_width`` as FWHM in km/s, ``fit_peak``, ``fit_line_flux`` and
``fit_continuum`` at the line centre, each with an ``_error``) are intrinsic, PB-corrected values.
Errors are formal, ``2 H^-1``, corrected for the channel correlation measured on the jackknife by
a sandwich estimator over the channels. ``meta["spectra"]`` keeps every fitted line's
PB-corrected spectrum and best fit, for :func:`~luv_finder.plotting.line_spectra_check`;
:func:`~luv_finder.imaging.line_maps` makes their moment-0 maps.

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
