"""Diagnostic figures shared by the CLI and the test suite.

Every function takes an output directory and returns the path it wrote, so the
figures produced by ``pytest --plots`` and by the command-line tools are the same
plots with the same styling.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, plots_dir: str, name: str) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name if name.endswith(".png") else name + ".png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def spectrum_check(data, dra, ddec, model=None, plots_dir="plots", name="spectrum", line_ghz=None):
    """Real and imaginary visibility spectra: data, jackknife and model.

    The data are shown at the reference direction and phase-shifted onto (``dra``, ``ddec``);
    the jackknife and the model (a source component) only shifted. A real source is flat at
    the reference and peaks once shifted; the jackknife should stay consistent with zero.
    """
    freqs = data.freqs / 1e9
    curves = [
        ("data, phase centre", data.spectrum(), {"c": "C0", "lw": 1, "alpha": 0.6}),
        ("data, shifted", data.spectrum(dra, ddec), {"c": "C1", "lw": 1.6}),
        ("jackknife, shifted", data.jackknife().spectrum(dra, ddec), {"c": "C7", "lw": 1, "ls": ":"}),
    ]
    if model is not None:
        curves.append(("model, shifted", data.spectrum(dra, ddec, model), {"c": "C2", "lw": 1.2, "alpha": 0.8}))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.5))
    for ax, part, label in zip(axes, (np.real, np.imag), ("Re", "Im"), strict=True):
        ax.axhline(0, c="gray", ls="--", lw=0.8)
        for curve, spectrum, style in curves:
            ax.plot(freqs, part(spectrum), label=curve, **style)
        if line_ghz is not None:
            ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
        ax.set_ylabel(f"{label}(V)  [Jy]")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Frequency [GHz]")
    axes[0].set_title(f'Visibility spectrum at dra={dra:+.2f}", ddec={ddec:+.2f}"')
    return _save(fig, plots_dir, name)


def response_check(mf, plots_dir="plots", name="filter_response", line_ghz=None):
    """Matched-filter response of the best grid point, with the jackknife overlaid.

    The y axis is signal-to-noise, so the peak height is the line's S/N.
    """
    freqs = mf.frequencies()
    best = mf.response[mf.best_index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, ls="--", c="gray", lw=0.8)
    for level in (3, 5):
        ax.axhline(level, ls=":", c="C3", lw=0.8)
        ax.text(freqs[0], level, f" {level}$\\sigma$", va="bottom", fontsize=7, c="C3")
    if line_ghz is not None:
        ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
    ax.plot(freqs, best, lw=1.8, label="data")
    if mf.response_jackknife is not None:
        ax.plot(freqs, mf.response_jackknife[mf.best_index], lw=1.2, ls=":", c="C7", label="jackknife")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Matched-filter S/N")
    ax.set_title(f"Peak S/N {np.nanmax(best):.1f} at {freqs[np.nanargmax(best)]:.3f} GHz")
    ax.legend(fontsize=8)
    text = "\n".join(f"{k.split('_', 2)[-1]}={v:.3g}" for k, v in mf.best_params.items())
    ax.text(
        1.02,
        0.5,
        text,
        transform=ax.transAxes,
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    return _save(fig, plots_dir, name)


def source_size_check(freqs, expected, noisy, declared, title, plots_dir="plots", name="source_size"):
    """Response at the true source position: noiseless expectation and one noisy draw.

    ``declared`` is the S/N the mock was calibrated to; a filter matched to the
    source reaches it with the expectation curve.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, ls="--", c="gray", lw=0.8)
    ax.axhline(declared, ls=":", c="C3", lw=1, label=f"declared S/N {declared:g}")
    ax.plot(freqs, expected, lw=1.8, label=f"expected (peak {np.max(expected):.2f})")
    ax.plot(freqs, noisy, lw=1, alpha=0.8, label=f"noisy draw (peak {np.max(noisy):.2f})")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Matched-filter S/N")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, plots_dir, name)


def responses_check(freqs, curves, title, plots_dir="plots", name="responses", line_ghz=None):
    """Several S/N spectra on one axis, e.g. the phase centre against the source positions.

    ``curves`` maps a legend label to a response with one value per frequency in ``freqs`` (GHz).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, ls="--", c="gray", lw=0.8)
    ax.axhline(5, ls=":", c="C3", lw=0.8)
    if line_ghz is not None:
        ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
    for label, response in curves.items():
        ax.plot(freqs, response, lw=1.4, label=f"{label} (peak {np.max(response):.1f})")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Matched-filter S/N")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, plots_dir, name)


def _nanmax(a, axis=None):
    """Maximum ignoring NaN; NaN where everything is NaN, without np.nanmax's warning."""
    return np.fmax.reduce(a, axis=axis)


def _sky_axes(ax):
    """East to the left, as on the sky."""
    ax.set_aspect("equal")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel("dra [arcsec east]")
    ax.set_ylabel("ddec [arcsec north]")


def _mark(ax, marks):
    for label, (dra, ddec) in (marks or {}).items():
        ax.plot(dra, ddec, "o", ms=14, mfc="none", mec="C3", mew=1.5)
        ax.annotate(label, (dra, ddec), xytext=(8, 8), textcoords="offset points", fontsize=8, c="C3")


def snr_map(mf, plots_dir="plots", name="snr_map", marks=None):
    """Peak S/N over channels, widths and sizes at every (dra, ddec) of the grid.

    The data and, when it was run, the jackknife are shown side by side on one colour scale.
    ``marks`` maps a label to a (dra, ddec) position to circle, e.g. an injected source. Positions
    where the response is NaN (outside every primary beam of a mosaic) are left blank.
    """
    dra, ddec = np.asarray(mf.axes["dra"]), np.asarray(mf.axes["ddec"])
    panels = {"data": mf.response}
    if mf.response_jackknife is not None:
        panels["jackknife"] = mf.response_jackknife
    peaks = {k: _nanmax(r.reshape(len(dra), len(ddec), -1), axis=-1) for k, r in panels.items()}
    vmax = max(_nanmax(p) for p in peaks.values())
    fig, axes = plt.subplots(1, len(peaks), figsize=(5.2 * len(peaks), 4.6), squeeze=False)
    for ax, (label, peak) in zip(axes[0], peaks.items(), strict=True):
        mesh = ax.pcolormesh(dra, ddec, peak.T, shading="nearest", vmin=0, vmax=vmax, cmap="magma")
        fig.colorbar(mesh, ax=ax, label="peak S/N")
        _mark(ax, marks)
        _sky_axes(ax)
        ax.set_title(f"{label}: max {_nanmax(peak):.1f}")
    return _save(fig, plots_dir, name)


def mosaic_layout(offsets, primary_beam, plots_dir="plots", name="mosaic", marks=None):
    """Pointing centres and half-power primary beams in the dataset's sky frame.

    ``offsets`` maps a field id to its phase centre (dra, ddec) in arcsec; ``primary_beam`` is the
    beam FWHM in arcsec.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for field, (dra, ddec) in offsets.items():
        ax.add_patch(plt.Circle((dra, ddec), primary_beam / 2, fill=False, lw=1, color="C0", alpha=0.7))
        ax.text(dra, ddec, str(field), ha="center", va="center", fontsize=10, color="C0")
    _mark(ax, marks)
    extent = np.abs(np.array(list(offsets.values()))).max() + primary_beam / 2
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    _sky_axes(ax)
    ax.set_title(f'{len(offsets)} pointings, primary beam FWHM {primary_beam:.0f}"')
    return _save(fig, plots_dir, name)


def noise_check(mf, plots_dir="plots", name="noise"):
    """Response statistics of the data against its jackknife and a unit Gaussian.

    Left: the distribution of the response over all grid points and channels. Right: how many
    of them exceed a threshold, which is what a detection threshold has to be calibrated on.
    Neighbouring grid points and channels are correlated, so the counts are not independent.
    NaN entries (positions outside every primary beam of a mosaic) are dropped.
    """
    from scipy.stats import norm

    panels = {"data": mf.response, "jackknife": mf.response_jackknife}
    panels = {k: v[np.isfinite(v)] for k, v in panels.items() if v is not None}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-7, 7, 141)
    for i, (label, values) in enumerate(panels.items()):
        left.hist(values, bins=bins, density=True, histtype="step", lw=1.4, color=f"C{i}",
                  label=f"{label}: std {values.std():.2f}")  # fmt: skip
        thresholds = np.linspace(0, 8, 81)
        right.semilogy(thresholds, [(values > t).sum() for t in thresholds], color=f"C{i}", label=label)
    x = np.linspace(-7, 7, 300)
    left.plot(x, norm.pdf(x), "k--", lw=1, label="unit Gaussian")
    left.set_yscale("log")
    left.set_ylim(1e-6, 1)
    left.set_xlabel("Matched-filter S/N")
    left.set_ylabel("Density")
    left.legend(fontsize=8)
    n = next(iter(panels.values())).size
    right.semilogy(thresholds, n * norm.sf(thresholds), "k--", lw=1, label="unit Gaussian, independent")
    right.set_ylim(0.5, n)
    right.set_xlabel("Threshold [S/N]")
    right.set_ylabel("Grid points x channels above")
    right.legend(fontsize=8)
    return _save(fig, plots_dir, name)


def uv_profile(datasets, dra, ddec, models, freq_ghz, plots_dir="plots", name="uv_profile", bins=25):
    """Visibility amplitude against uv distance at one channel, data binned and model envelopes.

    ``datasets`` and ``models`` map a label to a single-window :class:`~luv_finder.data.DataHandler`
    and a source component. Each dataset is phase-shifted onto (``dra``, ``ddec``) and scaled by
    its least-squares flux against the model envelope, so only the shapes are compared: a
    resolved source fades on long baselines.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, data) in enumerate(datasets.items()):
        (chunk,) = data.chunks
        k = int(np.argmin(np.abs(chunk.freq / 1e9 - freq_ghz)))
        one = chunk.channels(slice(k, k + 1))
        uvdist = np.hypot(*one.uv_waves())[0] / 1e3
        vis = (one.vis * one.phase(dra, ddec)).real[0]
        envelope = models[label].envelope(one)[0]
        w = one.w[0]
        flux = np.sum(w * vis * envelope) / np.sum(w * envelope**2)
        edges = np.linspace(0, uvdist.max(), bins + 1)
        idx = np.digitize(uvdist, edges) - 1
        sums = np.bincount(idx, w * vis / flux, bins + 1)[:bins]
        wsum = np.bincount(idx, w, bins + 1)[:bins]
        centres = 0.5 * (edges[1:] + edges[:-1])
        ok = wsum > 0
        ax.plot(centres[ok], sums[ok] / wsum[ok], "o", ms=4, color=f"C{i}", label=f"{label}, data")
        order = np.argsort(uvdist)
        ax.plot(uvdist[order], envelope[order], "-", lw=1.2, color=f"C{i}", alpha=0.8, label=f"{label}, model")
    ax.axhline(0, c="gray", ls="--", lw=0.8)
    ax.set_xlabel("uv distance [klambda]")
    ax.set_ylabel("Re(V) / flux")
    ax.set_title(f"Visibility profile at {freq_ghz:.3f} GHz")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, plots_dir, name)


def scan_check(x, curves, xlabel, title, truth=None, plots_dir="plots", name="scan"):
    """Peak S/N against one trial parameter, e.g. template size, width or position angle.

    ``curves`` maps a label to the peak S/N at every value of ``x``; ``truth`` marks the
    injected value.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in curves.items():
        ax.plot(x, y, "o-", ms=3, lw=1.4, label=label)
    if truth is not None:
        ax.axvline(truth, c="C3", ls="--", lw=0.8, label="injected")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak S/N")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, plots_dir, name)


def amp_phase_check(data, positions, plots_dir="plots", name="amp_phase", line_ghz=None):
    """Amplitude and phase of the vector-averaged visibility against frequency.

    ``positions`` maps a label to a (dra, ddec) the visibilities are phase-shifted onto first.
    On a source the phase stays near zero across its line and the amplitude rises; elsewhere the
    phase wanders and the amplitude is the noise floor.
    """
    freqs = data.freqs / 1e9
    fig, (amp, phase) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6))
    for label, (dra, ddec) in positions.items():
        spectrum = data.spectrum(dra, ddec)
        amp.plot(freqs, 1e3 * np.abs(spectrum), lw=1.2, label=label)
        phase.plot(freqs, np.degrees(np.angle(spectrum)), ".", ms=4, label=label)
    for ax in (amp, phase):
        if line_ghz is not None:
            ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
    phase.axhline(0, c="gray", ls="--", lw=0.8)
    phase.set_ylim(-185, 185)
    phase.set_yticks([-180, -90, 0, 90, 180])
    amp.set_ylabel("|V|  [mJy]")
    phase.set_ylabel("phase [deg]")
    phase.set_xlabel("Frequency [GHz]")
    amp.legend(fontsize=8)
    amp.set_title("Vector-averaged visibility")
    return _save(fig, plots_dir, name)


def dirty_maps_check(dra, ddec, moment8, continuum, sigma, plots_dir="plots", name="dirty_maps", marks=None):
    """Moment-8 (S/N), continuum (mJy/beam), and the moment-8 with continuum contours.

    Takes the output of :func:`luv_finder.matchedfilter.dirty_maps` on the ``dra x ddec`` grid.
    Contours are at -3 (dashed) and 3, 5, 10, 20, 50 times the continuum noise ``sigma``, a scalar
    or a per-pixel map shaped like ``continuum`` (a mosaic's noise varies across the field).
    NaN pixels, outside every primary beam, are left blank. The colour scales are robust to a few
    extreme pixels: continuum 1st to 99th percentile of the well-covered area (noise within twice
    its minimum), moment-8 0 to 99.9th percentile.
    """
    dra, ddec = np.asarray(dra), np.asarray(ddec)
    noise = f"noise {1e6 * sigma:.1f}" if np.ndim(sigma) == 0 else f"noise from {1e6 * np.nanmin(sigma):.1f}"
    snr = continuum / sigma
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    continuum_mjy = 1e3 * continuum
    moment8_limits = (0, np.nanpercentile(moment8, 99.9))
    covered = np.where(sigma <= 2 * np.nanmin(sigma), continuum_mjy, np.nan)
    continuum_limits = np.nanpercentile(covered, [1, 99])
    panels = (
        (axes[0], moment8, "moment-8 (line S/N)", "S/N", "magma", moment8_limits),
        (axes[1], continuum_mjy, f"continuum, {noise} uJy/beam", "mJy/beam", "viridis", continuum_limits),
        (axes[2], moment8, "moment-8 + continuum contours", "S/N", "magma", moment8_limits),
    )
    for ax, image, title, unit, cmap, (vmin, vmax) in panels:
        mesh = ax.pcolormesh(dra, ddec, image.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label=unit)
        ax.set_title(title)
        _mark(ax, marks)
        _sky_axes(ax)
    axes[2].contour(dra, ddec, snr.T, levels=[3, 5, 10, 20, 50], colors="w", linewidths=0.8)
    axes[2].contour(dra, ddec, snr.T, levels=[-3], colors="w", linewidths=0.8, linestyles="--")
    return _save(fig, plots_dir, name)


def pointings_check(freqs_ghz, pointings, combined, beam, plots_dir="plots", name="pointings", line_ghz=None):
    """One position seen through several pointings of a mosaic, and the combination.

    ``pointings`` maps a label to a dict with ``distance`` (arcsec from that pointing's centre),
    ``pb`` (primary beam at the line) and ``flux`` [Jy, primary-beam corrected], ``error`` [Jy]
    and ``snr`` over ``freqs_ghz`` (NaN where the pointing does not cover the position);
    ``combined`` has ``flux``, ``error`` and ``snr``; ``beam`` is (offsets, response) of the
    primary beam at the line. Left: where each pointing sits on the beam, hollow if it never
    covers the position. Middle: flux density with the combined ``+-`` 1 sigma band. Right: S/N,
    which the beam correction leaves unchanged. Pointings that never cover are left out of both.
    """
    freqs = np.asarray(freqs_ghz)
    unused = {label for label, p in pointings.items() if np.isnan(p["flux"]).all()}
    fig, (beam_ax, flux_ax, snr_ax) = plt.subplots(1, 3, figsize=(16, 4.5))
    beam_ax.plot(*beam, c="k", lw=1.2)
    for i, (label, p) in enumerate(pointings.items()):
        face = "none" if label in unused else f"C{i}"
        beam_ax.plot(p["distance"], p["pb"], "o", c=f"C{i}", markerfacecolor=face)
        text = f"{label} (not used)" if label in unused else label
        beam_ax.annotate(text, (p["distance"], p["pb"]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    beam_ax.set_xlabel("Offset from pointing centre [arcsec]")
    beam_ax.set_ylabel("Primary beam response")
    beam_ax.set_ylim(0, 1.05)
    beam_ax.set_title("Primary beam" if line_ghz is None else f"Primary beam at {line_ghz:.3f} GHz")

    flux_ax.axhline(0, ls="--", c="gray", lw=0.8)
    snr_ax.axhline(0, ls="--", c="gray", lw=0.8)
    snr_ax.axhline(5, ls=":", c="C3", lw=0.8)
    curves = {**pointings, "combined": combined}
    for ax, key, scale, fmt in ((flux_ax, "flux", 1e3, "{:.3g} mJy"), (snr_ax, "snr", 1, "{:.1f}")):
        for i, (label, p) in enumerate(curves.items()):
            if label in unused:
                continue
            y = scale * np.asarray(p[key])
            style = dict(c="k", lw=2.2) if p is combined else dict(c=f"C{i}", lw=1)
            ax.plot(freqs, y, label=f"{label} (peak {fmt.format(_nanmax(y))})", **style)
        if line_ghz is not None:
            ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
        ax.set_xlabel("Frequency [GHz]")
        ax.legend(fontsize=7)
    flux, error = (1e3 * np.asarray(combined[k]) for k in ("flux", "error"))
    flux_ax.fill_between(freqs, flux - error, flux + error, color="k", alpha=0.2)
    flux_ax.set_ylabel("Primary-beam corrected flux density [mJy]")
    snr_ax.set_ylabel("S/N")
    flux_ax.set_title(r"Flux density, combined $\pm 1\sigma$")
    snr_ax.set_title("S/N per pointing and combined")
    return _save(fig, plots_dir, name)


def _templates(result):
    """Name parts, shape (bmin, bmaj, pa) and line width [km/s] of every template of a search.

    The shape is named only when the search has more than one.
    """
    from itertools import product

    from .matchedfilter import TEMPLATE_KEYS

    combos = list(product(*(result.axes[k] for k in TEMPLATE_KEYS)))
    shapes = [c[:-1] for c in combos]
    many = len(set(shapes)) > 1
    parts = [([f'{bmaj:g}" x {bmin:g}" pa {pa:g}'] if many else []) + [f"{w:g} km/s"] for bmin, bmaj, pa, w in combos]
    return parts, shapes, np.array([c[-1] for c in combos])


def response_shape_check(nc, result, peaks=(), plots_dir="plots", name="response_shape"):
    """The null correlation rho of the response against what bright lines actually look like.

    ``nc`` is a :class:`~luv_finder.catalogue.NoiseCorrelation` measured on ``result``,
    ``peaks`` a sequence of ``(i, j, t, k)`` indices into ``result.snr``. Top row: rho over position
    lags for the first template with its half-power ellipse and rho = 0 contour; rho over channel
    lags per template (solid) against the white-channel analytic ``exp(-k^2 / 4 s^2)`` of a Gaussian
    template (dashed); rho between templates with, for templates differing only in width, the
    analytic ``sqrt(2 W1 W2 / (W1^2 + W2^2))`` in parentheses. Bottom row (if ``peaks``): cuts of
    ``snr / snr_peak`` through each peak along dra, ddec and channels, against the expectation
    ``E[snr(p)] = (S/N)_peak rho(p - p_peak)`` (thick, per template), within the lags of ``nc``.
    """
    from matplotlib.patches import Ellipse

    from .model import C_KMS, FWHM_TO_SIGMA

    parts, shapes, widths = _templates(result)
    names, n_t = [", ".join(p) for p in parts], len(parts)
    nu, dnu = np.median(result.freqs), np.median(np.abs(np.diff(result.freqs)))
    chan_width = nu * widths / C_KMS / dnu
    fig, axes = plt.subplots(2 if len(peaks) else 1, 3, figsize=(16, 4.8 * (2 if len(peaks) else 1)), squeeze=False)
    sky, spectral, matrix = axes[0]

    mesh = sky.pcolormesh(nc.lags_ra, nc.lags_dec, nc.spatial[0].T, shading="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(mesh, ax=sky, label=r"$\rho$")
    sky.contour(nc.lags_ra, nc.lags_dec, nc.spatial[0].T, levels=[0], colors="k", linewidths=0.8)
    major, minor = nc.hwhm_major[0], nc.hwhm_minor[0]
    sky.add_patch(Ellipse((0, 0), 2 * major, 2 * minor, angle=90 - nc.pa[0], fill=False, ec="k", ls="--", lw=1.5))
    _sky_axes(sky)
    sky.set_title(f'{names[0]}: HWHM {major:.2f}" x {minor:.2f}"\nPA {nc.pa[0]:.0f} deg, sidelobe {nc.sidelobe[0]:.2f}')

    lag = np.arange(nc.spectral.shape[1])
    fine = np.linspace(0, lag[-1], 100)
    spectral.plot([], [], "k--", lw=1, label="white channels, analytic")
    for t in range(n_t):
        sigma = chan_width[t] * FWHM_TO_SIGMA
        hwhm = f"HWHM {nc.spectral_hwhm[t]:.2f} ch = {nc.spectral_hwhm_kms[t]:.0f} km/s"
        spectral.plot(lag, nc.spectral[t], "o-", ms=3, c=f"C{t}", label=f"{names[t]}: {hwhm}")
        spectral.plot(fine, np.exp(-(fine**2) / (4 * sigma**2)), "--", lw=1, c=f"C{t}")
    spectral.axhline(0, c="gray", ls="--", lw=0.8)
    spectral.set_xlabel("Channel lag")
    spectral.set_ylabel(r"$\rho$")
    spectral.set_title("Spectral correlation")
    spectral.legend(fontsize=8)

    matrix.imshow(nc.templates, cmap="RdBu_r", vmin=-1, vmax=1)
    for i, j in np.ndindex(n_t, n_t):
        text = f"{nc.templates[i, j]:.2f}"
        if i != j and shapes[i] == shapes[j]:
            w1, w2 = widths[[i, j]]
            text += f"\n({np.sqrt(2 * w1 * w2 / (w1**2 + w2**2)):.2f})"
        color = "w" if abs(nc.templates[i, j]) > 0.75 else "k"
        matrix.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ticks = ["\n".join(p) for p in parts]
    matrix.set_xticks(range(n_t), ticks, fontsize=8)
    matrix.set_yticks(range(n_t), ticks, fontsize=8)
    matrix.set_title("Template correlation (analytic in parentheses)")

    if len(peaks):
        cuts = axes[1]
        centre = (len(nc.lags_ra) // 2, len(nc.lags_dec) // 2)
        lags = (nc.lags_ra, nc.lags_dec, np.arange(1 - len(lag), len(lag)))
        axis_dra, axis_ddec = np.asarray(result.axes["dra"]), np.asarray(result.axes["ddec"])
        chan = np.arange(result.snr.shape[-1])
        colors = plt.cm.plasma(np.linspace(0, 0.7, len(peaks)))
        for color, (i, j, t, k) in zip(colors, peaks, strict=True):
            peak = result.snr[i, j, t, k]
            offsets = (axis_dra - axis_dra[i], axis_ddec - axis_ddec[j], chan - k)
            profiles = (result.snr[:, j, t, k], result.snr[i, :, t, k], result.snr[i, j, t, :])
            label = f"S/N {peak:.1f}, {result.freqs[k] / 1e9:.3f} GHz" + (f", {names[t]}" if n_t > 1 else "")
            for ax, x, y, reach in zip(cuts, offsets, profiles, lags, strict=True):
                near = np.abs(x) <= reach.max() * (1 + 1e-6)
                ax.plot(x[near], y[near] / peak, lw=1, color=color, label=label, zorder=2)
        for n, t in enumerate(sorted({p[2] for p in peaks})):
            expected = (
                nc.spatial[t][:, centre[1]],
                nc.spatial[t][centre[0]],
                np.r_[nc.spectral[t][:0:-1], nc.spectral[t]],
            )
            style = ("-", "--", ":")[n % 3]
            label = r"expected $\rho$" + (f", {names[t]}" if n_t > 1 else "")
            for ax, x, y in zip(cuts, lags, expected, strict=True):
                ax.plot(x, y, c="k", ls=style, lw=3, alpha=0.5, label=label, zorder=1)
        labels = ("dra offset from the peak [arcsec east]", "ddec offset from the peak [arcsec north]", "channel lag")
        for ax, reach, label in zip(cuts, lags, labels, strict=True):
            ax.axhline(0, c="gray", ls="--", lw=0.8)
            ax.set_xlim(reach.min(), reach.max())
            ax.set_xlabel(label)
            ax.set_ylabel("S/N / peak S/N")
            ax.set_title(f"Cut along {label.split()[0]}")
        cuts[2].legend(fontsize=7, loc="upper left")
    return _save(fig, plots_dir, name)


def _count_above(values, at):
    """Number of ``values`` (sorted) at or above each of ``at``."""
    return len(values) - np.searchsorted(values, at)


def reliability_check(cat, plots_dir="plots", name="reliability"):
    """Counts, likelihood ratio and fidelity of a catalogue against the jackknife, its noise reference.

    ``cat`` is the output of :func:`luv_finder.catalogue.catalogue` (its ``meta`` holds what is
    plotted, so a subset of its rows works too). Left: cumulative group counts N(>= S/N) of the
    data, the jackknife and the negated data (a diagnostic only), with the detected rows on the
    data curve. Middle: the likelihood ratio N_data / N_jackknife at every data group, a lower
    limit (caret) where no jackknife group is as bright, against the threshold ``k`` and the
    lowest S/N above it. Right: the binned fidelity ``1 - N_jackknife / N_data`` with its erf fit
    and the S/N where that reaches ``fidelity_min``. The fit and the thresholds are left out where
    they are NaN.
    """
    from matplotlib.ticker import LogFormatter

    from .catalogue import fidelity_curve, likelihood

    meta = cat.meta
    floor, k, fidelity_min = meta["floor"], meta["k"], meta["fidelity_min"]
    data, jackknife, negative = (np.sort(meta[f"snr_{key}"]) for key in ("data", "jackknife", "negative"))
    detected = np.asarray(cat["snr"])[np.asarray(cat["detected"], dtype=bool)]
    xlim = (floor, data.max(initial=floor) + 0.5)

    fig, (counts, ratio, fid) = plt.subplots(1, 3, figsize=(16, 4.5))
    styles = {"data": dict(c="k"), "jackknife": dict(c="C0"), "negated data": dict(c="0.5", ls="--")}
    for (label, style), values in zip(styles.items(), (data, jackknife, negative), strict=True):
        n = np.r_[len(values), _count_above(values, values)]
        shown = n > 0
        counts.step(np.r_[floor, values][shown], n[shown], where="pre", label=f"{label} ({len(values)})", **style)
    counts.plot(detected, _count_above(data, detected), "o", c="C3", ms=6, label=f"detected ({detected.size})")
    counts.set_ylim(bottom=0.7)
    counts.set_ylabel(r"Groups with S/N $\geq$ abscissa")

    value, lower = likelihood(data, jackknife)
    for is_limit, marker, label in ((False, "o", r"$\Lambda$"), (True, "^", "lower limit")):
        use = lower == is_limit
        ratio.plot(data[use], value[use], marker, c="k", mfc="none" if is_limit else "k", ms=5, ls="", label=label)
    ratio.axhline(k, c="C3", ls="--", lw=1, label=f"k = {k:g}")
    ratio.set_ylabel(r"Likelihood ratio $\Lambda$ = N$_{data}$ / N$_{jackknife}$")

    bins = meta["fidelity_bins"]
    fid.errorbar(
        bins["snr"], bins["fidelity"], bins["error"], fmt="o", c="k", ecolor="0.6", ms=4, capsize=2, label="binned"
    )
    centre, sigma = meta["fidelity_centre"], meta["fidelity_sigma"]
    if np.isfinite([centre, sigma]).all():
        x = np.linspace(*xlim, 200)
        fid.plot(
            x, fidelity_curve(x, centre, sigma), c="C1", label=rf"erf fit, C = {centre:.2f}, $\sigma$ = {sigma:.2f}"
        )
    fid.axhline(fidelity_min, c="C3", ls="--", lw=1, label=f"fidelity_min = {fidelity_min:g}")
    fid.set_ylim(-0.05, 1.05)
    fid.set_ylabel(r"Fidelity 1 $-$ N$_{jackknife}$ / N$_{data}$")

    for ax, key in ((ratio, "snr_likelihood"), (fid, "snr_fidelity")):
        if np.isfinite(meta[key]):
            ax.axvline(meta[key], c="C3", ls=":", lw=1, label=f"threshold at S/N {meta[key]:.2f}")
    for ax, title in zip((counts, ratio, fid), ("Cumulative counts", "Likelihood ratio", "Fidelity"), strict=True):
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_xlabel("S/N")
    for ax in (counts, ratio):
        ax.set_yscale("log")
        plain = LogFormatter(minor_thresholds=(2, 0.5))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.legend(fontsize=8)
    fid.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle(f"{detected.size} of {len(data)} data groups detected")
    return _save(fig, plots_dir, name)


def contact_sheet(plots_dir: str, title: str = "Luv_finder diagnostics") -> str:
    """Write an index.html showing every PNG in ``plots_dir``."""
    pngs = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
    cards = "\n".join(f'<figure><img src="{f}" loading="lazy"><figcaption>{f[:-4]}</figcaption></figure>' for f in pngs)
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}}
 h1{{font-size:1.2rem}} .grid{{display:grid;gap:1.5rem;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}}
 figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:.75rem}}
 img{{width:100%;height:auto}}
 figcaption{{margin-top:.5rem;font-family:ui-monospace,monospace;font-size:12px;color:#555}}
</style>
<h1>{title}</h1><p>{len(pngs)} figures</p><div class="grid">{cards}</div>"""
    path = os.path.join(plots_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path
