Catalogue and line fit
======================

Catalogue
---------

:func:`luv_finder.catalogue.catalogue` turns a search with its jackknife (one pointing
or the combined mosaic, on a uniform position lattice) into an astropy table of line
candidates. The jackknife, which has the data's noise and no sky, is the reference
throughout:

1. the S/N is the filter's, normalised by its exact variance under the correlation
   between channels measured on the jackknife per spectral window
   (:func:`~luv_finder.data.channel_correlation`), so it has unit variance under the null
   (:func:`~luv_finder.catalogue.jackknife_spread` checks it);
2. the correlation function of the jackknife response is measured; it is also the
   expected shape of a matched line's response (Vio & Andreani 2021), so its half-power
   ellipsoid in position and frequency is what one source occupies;
3. local maxima of the best-template S/N above 4 are merged, brightest first, within that
   ellipsoid, for the data, the jackknife and (as a diagnostic) the negated data;
4. every data group gets the likelihood ratio ``N_data(>= S/N) / N_jackknife(>= S/N)``
   (van Marrewijk et al. 2025, required >= 3) and the fidelity ``1 - N_jackknife / N_data``
   per S/N bin, fitted with an error function (Walter et al. 2016, required >= 0.6).

.. code-block:: python

   from luv_finder.catalogue import catalogue

   cat = catalogue(result, ref=data.metadata.ref)
   lines = cat[cat["detected"]]
   cat.write("lines.ecsv", format="ascii.ecsv")

or ``luv-find ... --jackknife --catalogue lines.ecsv``.

Fitting the detected lines
--------------------------

Detection stays a grid search. :func:`luv_finder.fit.fit_lines` then refines every
detected line by least squares in the visibilities: an elliptical Gaussian source
(position and sky covariance) whose spectrum is a Gaussian line on a polynomial continuum,
times each pointing's primary beam at the source, jointly in every pointing whose primary
beam covers the line (PB >= 0.2), in the spectral window holding it.

.. code-block:: python

   from luv_finder.fit import fit_lines

   fitted = fit_lines("mosaic.npz", cat)  # or the search's DataHandler
   fitted.write("lines.ecsv", format="ascii.ecsv", overwrite=True)

or ``luv-find ... --jackknife --catalogue lines.ecsv --fit``. The fitted columns
(``fit_dra``, ``fit_ddec``, ``fit_ra``, ``fit_dec``; ``fit_bmaj``, ``fit_bmin`` as FWHM
and ``fit_pa``; ``fit_freq_ghz``, ``fit_width`` as FWHM in km/s, ``fit_peak``,
``fit_line_flux`` and ``fit_continuum`` at the line centre, each with an ``_error``) are
intrinsic, PB-corrected values. An unresolved axis lands on zero size, with a NaN error.
Errors are formal, corrected for the channel correlation. ``meta["spectra"]`` keeps every
fitted line's spectrum and best fit; :func:`~luv_finder.imaging.line_maps` makes their
moment-0 maps.

Diagnostic figures
------------------

:mod:`luv_finder.plotting` holds every figure; each takes a directory and returns the
path written. :func:`~luv_finder.plotting.reliability_check` shows the counts, the
likelihood ratio and the fidelity; :func:`~luv_finder.plotting.response_shape_check` the
response correlation against real lines; :func:`~luv_finder.plotting.line_spectra_check`
and :func:`~luv_finder.plotting.line_maps_check` the fitted lines. ``pytest --plots``
writes the test figures to ``plots/`` with an ``index.html`` contact sheet.
