"""Diagnostic figures shared by the CLI, the run products and the test suite.

Every function returns its :class:`~matplotlib.figure.Figure` and writes ``<plots_dir>/<name>.png`` only
when ``plots_dir`` is given, so a notebook shows the figure and a script or ``pytest --plots`` saves the
same plot. Figures are built without pyplot (nothing to close, never shown twice in a notebook) and
inside :func:`style`, the package's matplotlib style, which leaves the caller's rcParams alone.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

from .model import C_KMS

#: The package's matplotlib style file.
STYLE = Path(__file__).with_name("luv.mplstyle")


def style():
    """The Luv_finder figure style as a context manager: ``with style(): ...``."""
    return plt.style.context(STYLE)


def _styled(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with style():
            return func(*args, **kwargs)

    return wrapper


def _subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols, **kwargs)


def _save(fig: Figure, plots_dir: str | None, name: str) -> Figure:
    fig.tight_layout()
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, name if name.endswith(".png") else name + ".png"))
    return fig


@_styled
def spectrum_check(data, dra, ddec, model=None, plots_dir=None, name="spectrum", line_ghz=None):
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

    fig, axes = _subplots(2, 1, sharex=True, figsize=(7.5, 6.5))
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


@_styled
def response_check(mf, plots_dir=None, name="filter_response", line_ghz=None):
    """Matched-filter response of the best grid point, with the jackknife overlaid.

    The y axis is signal-to-noise, so the peak height is the line's S/N.
    """
    freqs = mf.frequencies()
    best = mf.response[mf.best_index]
    fig, ax = _subplots(figsize=(8, 4))
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


@_styled
def responses_check(freqs, curves, title, plots_dir=None, name="responses", line_ghz=None):
    """Several S/N spectra on one axis, e.g. the phase centre against the source positions.

    ``curves`` maps a legend label to a response with one value per frequency in ``freqs`` (GHz).
    """
    fig, ax = _subplots(figsize=(8, 4))
    ax.axhline(0, ls="--", c="gray", lw=0.8)
    ax.axhline(5, ls=":", c="C3", lw=0.8)
    if line_ghz is not None:
        ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
    for label, response in curves.items():
        ax.plot(freqs, response, lw=1.4, label=label)
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


@_styled
def snr_map(mf, plots_dir=None, name="snr_map", marks=None):
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
    fig, axes = _subplots(1, len(peaks), figsize=(5.2 * len(peaks), 4.6), squeeze=False)
    for ax, (label, peak) in zip(axes[0], peaks.items(), strict=True):
        mesh = ax.pcolormesh(dra, ddec, peak.T, shading="nearest", vmin=0, vmax=vmax, cmap="magma")
        fig.colorbar(mesh, ax=ax, label="peak S/N")
        _mark(ax, marks)
        _sky_axes(ax)
        ax.set_title(f"{label}: max {_nanmax(peak):.1f}")
    return _save(fig, plots_dir, name)


@_styled
def mosaic_layout(offsets, primary_beam, plots_dir=None, name="mosaic", marks=None):
    """Pointing centres and half-power primary beams in the dataset's sky frame.

    ``offsets`` maps a field id to its phase centre (dra, ddec) in arcsec; ``primary_beam`` is the
    beam FWHM in arcsec.
    """
    fig, ax = _subplots(figsize=(5.5, 5.5))
    for field, (dra, ddec) in offsets.items():
        ax.add_patch(Circle((dra, ddec), primary_beam / 2, fill=False, lw=1, color="C0", alpha=0.7))
        ax.text(dra, ddec, str(field), ha="center", va="center", fontsize=10, color="C0")
    _mark(ax, marks)
    extent = np.abs(np.array(list(offsets.values()))).max() + primary_beam / 2
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    _sky_axes(ax)
    ax.set_title(f'{len(offsets)} pointings, primary beam FWHM {primary_beam:.0f}"')
    return _save(fig, plots_dir, name)


@_styled
def noise_check(mf, plots_dir=None, name="noise"):
    """Response statistics of the data against its jackknife and a unit Gaussian.

    Left: the distribution of the response over all grid points and channels. Right: how many
    of them exceed a threshold, which is what a detection threshold has to be calibrated on.
    Neighbouring grid points and channels are correlated, so the counts are not independent.
    NaN entries (positions outside every primary beam of a mosaic) are dropped.
    """
    from scipy.stats import norm

    panels = {"data": mf.response, "jackknife": mf.response_jackknife}
    panels = {k: v[np.isfinite(v)] for k, v in panels.items() if v is not None}
    fig, (left, right) = _subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-7, 7, 141)
    for i, (label, values) in enumerate(panels.items()):
        left.hist(values, bins=bins, density=True, histtype="step", lw=1.4, color=f"C{i}", label=label)
        thresholds = np.linspace(0, 8, 81)
        right.semilogy(thresholds, [(values > t).sum() for t in thresholds], color=f"C{i}", label=label)
    x = np.linspace(-7, 7, 300)
    left.plot(x, norm.pdf(x), "k--", lw=1, label="unit Gaussian")
    left.set_yscale("log")
    left.set_ylim(1e-6, 1)
    left.set_xlabel("Matched-filter S/N")
    left.set_ylabel("Density")
    left.set_title("std " + ", ".join(f"{label} {values.std():.2f}" for label, values in panels.items()))
    left.legend(fontsize=8)
    n = next(iter(panels.values())).size
    right.semilogy(thresholds, n * norm.sf(thresholds), "k--", lw=1, label="unit Gaussian, independent")
    right.set_ylim(0.5, n)
    right.set_xlabel("Threshold [S/N]")
    right.set_ylabel("Grid points x channels above")
    right.legend(fontsize=8)
    return _save(fig, plots_dir, name)


@_styled
def amp_phase_check(data, positions, plots_dir=None, name="amp_phase", line_ghz=None):
    """Amplitude and phase of the vector-averaged visibility against frequency.

    ``positions`` maps a label to a (dra, ddec) the visibilities are phase-shifted onto first.
    On a source the phase stays near zero across its line and the amplitude rises; elsewhere the
    phase wanders and the amplitude is the noise floor.
    """
    freqs = data.freqs / 1e9
    fig, (amp, phase) = _subplots(2, 1, sharex=True, figsize=(7.5, 6))
    for label, (dra, ddec) in positions.items():
        spectrum = data.spectrum(dra, ddec)
        amp.plot(freqs, 1e3 * np.abs(spectrum), lw=1.2, label=label)
        deg = np.degrees(np.angle(spectrum))
        wrap = np.flatnonzero(np.abs(np.diff(deg)) > 180) + 1  # no line across the +-180 wrap
        phase.plot(np.insert(freqs, wrap, np.nan), np.insert(deg, wrap, np.nan), ".-", ms=4, lw=0.8, label=label)
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


@_styled
def dirty_maps_check(dra, ddec, moment8, continuum, sigma, plots_dir=None, name="dirty_maps", marks=None):
    """Moment-8 (S/N), continuum (mJy/beam), and the moment-8 with continuum contours.

    Takes the output of :func:`luv_finder.imaging.dirty_maps` on the ``dra x ddec`` grid.
    Contours are at -3 (dashed) and 3, 5, 10, 20, 50 times the continuum noise ``sigma``, a scalar
    or a per-pixel map shaped like ``continuum`` (a mosaic's noise varies across the field).
    NaN pixels, outside every primary beam, are left blank. The colour scales are robust to a few
    extreme pixels: continuum 1st to 99th percentile of the well-covered area (noise within twice
    its minimum), moment-8 0 to 99.9th percentile.
    """
    dra, ddec = np.asarray(dra), np.asarray(ddec)
    noise = f"noise {1e6 * sigma:.1f}" if np.ndim(sigma) == 0 else f"noise from {1e6 * np.nanmin(sigma):.1f}"
    snr = continuum / sigma
    fig, axes = _subplots(1, 3, figsize=(16, 4.8))
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


@_styled
def pointings_check(freqs_ghz, pointings, combined, beam, plots_dir=None, name="pointings", line_ghz=None):
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
    fig, (beam_ax, flux_ax, snr_ax) = _subplots(1, 3, figsize=(16, 4.5))
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
    for ax, key, scale in ((flux_ax, "flux", 1e3), (snr_ax, "snr", 1)):
        for i, (label, p) in enumerate(curves.items()):
            if label in unused:
                continue
            y = scale * np.asarray(p[key])
            style = dict(c="k", lw=2.2) if p is combined else dict(c=f"C{i}", lw=1)
            ax.plot(freqs, y, label=label, **style)
        if line_ghz is not None:
            ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
        ax.set_xlabel("Frequency [GHz]")
        ax.legend(fontsize=7)
    flux, error = (1e3 * np.asarray(combined[k]) for k in ("flux", "error"))
    flux_ax.fill_between(freqs, flux - error, flux + error, color="k", alpha=0.2)
    flux_ax.set_ylabel("Primary-beam corrected flux density [mJy]")
    snr_ax.set_ylabel("S/N")
    flux_ax.set_title(r"Flux density, combined $\pm 1\sigma$")
    snr_ax.set_title(f"S/N per pointing and combined: peak {_nanmax(np.asarray(combined['snr'])):.1f}")
    return _save(fig, plots_dir, name)


def _templates(result):
    """Name parts, shape (bmin, bmaj, pa) and line width [km/s] of every template of a search.

    The shape is named only when the search has more than one.
    """
    combos = result.templates
    shapes = [tuple(c[:-1]) for c in combos]
    many = len(set(shapes)) > 1
    parts = [([f'{bmaj:g}" x {bmin:g}" pa {pa:g}'] if many else []) + [f"{w:g} km/s"] for bmin, bmaj, pa, w in combos]
    return parts, shapes, np.array([c[-1] for c in combos])


@_styled
def response_shape_check(nc, result, peaks=(), plots_dir=None, name="response_shape"):
    """The null correlation rho of the response against what bright lines actually look like.

    ``nc`` is a :class:`~luv_finder.catalogue.ResponseCorrelation` measured on ``result``,
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
    fig, axes = _subplots(2 if len(peaks) else 1, 3, figsize=(16, 4.8 * (2 if len(peaks) else 1)), squeeze=False)
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
        spectral.plot(lag, nc.spectral[t], "o-", ms=3, c=f"C{t}", label=names[t])
        spectral.plot(fine, np.exp(-(fine**2) / (4 * sigma**2)), "--", lw=1, c=f"C{t}")
    spectral.axhline(0, c="gray", ls="--", lw=0.8)
    spectral.set_xlabel("Channel lag")
    spectral.set_ylabel(r"$\rho$")
    spectral.set_title("Spectral correlation, HWHM " + ", ".join(f"{h:.0f}" for h in nc.spectral_hwhm_kms) + " km/s")
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


def erf_curve(snr, fidelity) -> tuple[float, float]:
    """``(centre, width)`` of ``F = (1 + erf((S/N - centre) / sqrt(2) width)) / 2`` fitted to fidelity points.

    For display only: it smooths the tail model's binned fidelity; the thresholds come from the bins.
    NaN when fewer than three points are finite or the fit fails.
    """
    from scipy import optimize, special

    snr, fidelity = np.asarray(snr, dtype=float), np.asarray(fidelity, dtype=float)
    use = np.isfinite(snr) & np.isfinite(fidelity)
    if use.sum() < 3:
        return np.nan, np.nan
    centre = snr[use][np.argmin(np.abs(fidelity[use] - 0.5))]
    try:
        p, _ = optimize.curve_fit(
            lambda x, c, w: 0.5 * (1 + special.erf((x - c) / (np.sqrt(2) * w))),
            snr[use],
            fidelity[use],
            p0=(centre, 0.5),
            bounds=([0.0, 0.01], [np.inf, 10.0]),
        )
    except RuntimeError:
        return np.nan, np.nan
    return float(p[0]), float(p[1])


def _erf_plot(ax, bins, xlim, **style):
    """The :func:`erf_curve` of the tail model's ``bins`` over ``xlim``, labelled with its centre and width."""
    from scipy import special

    centre, width = erf_curve(bins["snr"], bins["model"])
    if np.isfinite(centre):
        x = np.linspace(*xlim, 300)
        label = style.pop("label")
        ax.plot(x, 0.5 * (1 + special.erf((x - centre) / (np.sqrt(2) * width))),
                label=rf"{label} (C {centre:.2f}, $\sigma$ {width:.2f})", **style)  # fmt: skip


@_styled
def reliability_check(cat, plots_dir=None, name="reliability"):
    """Counts, likelihood ratio and fidelity of a catalogue against the jackknife, its noise reference.

    ``cat`` is the output of :func:`luv_finder.catalogue.catalogue` (its ``meta`` holds what is
    plotted, so a subset of its rows works too). Left: cumulative group counts N(>= S/N) of the
    data, the jackknife and the negated data (a diagnostic only), with the detected rows on the
    data curve, and the Gaussian tail fitted to the jackknife's. Middle: the likelihood ratio
    N_data / N_jackknife at every data group, a lower limit (caret) where no jackknife group is as
    bright, against the threshold ``k`` and the lowest S/N above it. Right: the binned fidelity
    ``1 - N_jackknife / N_data``, the fidelity from the jackknife's tail with an erf through it
    (:func:`erf_curve`, display only) and the lowest S/N reaching ``fidelity_min``, and both against
    the negated data (as ASPECS measures it) for comparison. The tails and the
    thresholds are left out where they are NaN.
    """
    from matplotlib.ticker import LogFormatter
    from scipy import special

    from .catalogue import fidelity, likelihood

    meta = cat.meta
    floor, k, fidelity_min = meta["floor"], meta["k"], meta["fidelity_min"]
    data, jackknife, negative = (np.sort(meta[f"snr_{key}"]) for key in ("data", "jackknife", "negative"))
    detected = np.asarray(cat["snr"])[np.asarray(cat["detected"], dtype=bool)]
    xlim = (floor, data.max(initial=floor) + 0.5)

    fig, (counts, ratio, fid) = _subplots(1, 3, figsize=(16, 4.5))
    styles = {"data": dict(c="k"), "jackknife": dict(c="C0"), "negated data": dict(c="0.5", ls="--")}
    for (label, style), values in zip(styles.items(), (data, jackknife, negative), strict=True):
        n = np.r_[len(values), _count_above(values, values)]
        shown = n > 0
        counts.step(np.r_[floor, values][shown], n[shown], where="pre", label=label, **style)
    counts.plot(detected, _count_above(data, detected), "o", c="C3", ms=6, label="detected")
    amplitude, sigma = meta["noise_amplitude"], meta["noise_sigma"]
    if np.isfinite([amplitude, sigma]).all():
        x = np.linspace(*xlim, 200)
        counts.plot(x, amplitude * special.erfc(x / (np.sqrt(2) * sigma)), c="C1", label="jackknife tail")
    counts.set_ylim(bottom=0.7)
    counts.set_ylabel(r"Groups with S/N $\geq$ abscissa")

    value, lower = likelihood(data, jackknife)
    for is_limit, marker, label in ((False, "o", r"$\Lambda$"), (True, "^", "lower limit")):
        use = lower == is_limit
        ratio.plot(data[use], value[use], marker, c="k", mfc="none" if is_limit else "k", ms=5, ls="", label=label)
    ratio.axhline(k, c="C3", ls="--", lw=1, label="k")
    ratio.set_ylabel(r"Likelihood ratio $\Lambda$ = N$_{data}$ / N$_{jackknife}$")

    bins = meta["fidelity_bins"]
    fid.errorbar(
        bins["snr"], bins["fidelity"], bins["error"], fmt="o", c="k", ecolor="0.6", ms=4, capsize=2, label="binned"
    )
    fid.plot(bins["snr"], bins["model"], ".", c="C1", label="jackknife tail")
    _erf_plot(fid, bins, xlim, c="C1", label="erf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, (_, s_neg), bins_neg = fidelity(data, negative, floor)
    fid.plot(bins_neg["snr"], bins_neg["fidelity"], "s", mfc="none", c="0.45", ms=4,
             label="binned, negated")  # fmt: skip
    if np.isfinite(s_neg):
        fid.plot(bins_neg["snr"], bins_neg["model"], ".", c="0.45", label="negated tail")
        _erf_plot(fid, bins_neg, xlim, c="0.45", ls="--", label="erf, negated")
    fid.axhline(fidelity_min, c="C3", ls="--", lw=1, label="fidelity_min")
    fid.set_ylim(-0.05, 1.05)
    fid.set_ylabel(r"Fidelity 1 $-$ N$_{jackknife}$ / N$_{data}$")

    for ax, key in ((ratio, "snr_likelihood"), (fid, "snr_fidelity")):
        if np.isfinite(meta[key]):
            ax.axvline(meta[key], c="C3", ls=":", lw=1, label="threshold")
    passing = data[values >= fidelity_min] if np.isfinite(s_neg) else np.array([])
    titles = (
        f"Counts: {len(data)} data, {len(jackknife)} jackknife, {len(negative)} negated"
        + (rf"; tail $\sigma$ {sigma:.3f}" if np.isfinite(sigma) else ""),
        rf"Likelihood ratio: $\Lambda \geq$ {k:g}"
        + (f" from S/N {meta['snr_likelihood']:.2f}" if np.isfinite(meta["snr_likelihood"]) else ""),
        f"Fidelity: {fidelity_min:g}"
        + (f" at S/N {meta['snr_fidelity']:.2f}" if np.isfinite(meta["snr_fidelity"]) else "")
        + (f" (negated {passing.min():.2f})" if passing.size else ""),
    )
    for ax, title in zip((counts, ratio, fid), titles, strict=True):
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


def _whitened(chunks):
    """``Re V sqrt(w)`` per channel over ``chunks`` of one window, unflagged: unit variance where the weights hold."""
    zs = [np.where(c.w > 0, c.vis.real * np.sqrt(c.w), np.nan) for c in chunks]
    return [x[np.isfinite(x)] for x in np.concatenate(zs, axis=1)]


def _noise_ratio(z):
    """Robust sigma per channel of whitened samples ``z`` and its counting error, ``1.166 / sqrt(N)`` for a MAD."""
    from scipy import stats

    sigma = np.array([stats.median_abs_deviation(x, scale="normal") if x.size else np.nan for x in z])
    return sigma, 1.166 / np.sqrt(np.maximum([x.size for x in z], 1))


def _ratio_line(ax, ghz, z, label, **style):
    ratio, error = _noise_ratio(z)
    (line,) = ax.plot(ghz, ratio, lw=1, label=label, **style)
    ax.fill_between(ghz, 1 - error, 1 + error, color=line.get_color(), alpha=0.2, lw=0)
    return ratio


@_styled
def weights_check(chunks, labels, plots_dir=None, name="weights", n_points=200_000, seed=0, pooled=()):
    """Do the weights predict the scatter? Re V at the phase centre per channel, its spread and ``<1/w>^1/2``.

    One panel per chunk (one field and window, e.g. with the measurement set's weights and reweighted), each
    with a random ``n_points`` of the visibilities, the measured spread (median and robust sigma per channel)
    and the spread the weights predict; below, the measured over the predicted spread per channel, from
    ``Re V sqrt(w)``, which is 1 where the weights hold. The sky is in V too, which is fine for a faint field.
    The ratio scatters by ``1.166 / sqrt(N)`` per channel (shaded), so one field ripples; ``pooled``, the
    window's chunks of every field, tests the reweighting, which is fitted per scan over them all.
    """
    from scipy import stats

    rng, ratios = np.random.default_rng(seed), []
    fig, axes = _subplots(
        len(chunks) + 1, 1, figsize=(11, 3.1 * len(chunks) + 2.6), sharex=True,
        gridspec_kw={"height_ratios": [3] * len(chunks) + [2]},
    )  # fmt: skip
    for ax, c, label in zip(axes[:-1], chunks, labels, strict=True):
        ghz, re, ok = c.freq / 1e9, c.vis.real, c.w > 0
        chan, row = np.nonzero(ok)
        pick = rng.choice(chan.size, min(n_points, chan.size), replace=False)
        ax.plot(ghz[chan[pick]], 1e3 * re[chan[pick], row[pick]], ",", c="0.6", rasterized=True,
                label="visibilities")  # fmt: skip
        centre = np.array([np.median(r[o]) if o.any() else np.nan for r, o in zip(re, ok, strict=True)])
        spread = np.array([stats.median_abs_deviation(r[o], scale="normal") if o.any() else np.nan
                           for r, o in zip(re, ok, strict=True)])  # fmt: skip
        predicted = np.sqrt([np.mean(1 / w[o]) if o.any() else np.nan for w, o in zip(c.w, ok, strict=True)])
        ax.fill_between(ghz, 1e3 * (centre - spread), 1e3 * (centre + spread), color="C0", alpha=0.35,
                        label="measured ±1σ")  # fmt: skip
        predicts = r"weights: $\pm\langle 1/w\rangle^{1/2}$"
        for sign, text in ((1, predicts), (-1, None)):
            ax.plot(ghz, 1e3 * sign * predicted, "--", c="C1", label=text)
        z = _whitened([c])
        overall = stats.median_abs_deviation(np.concatenate(z), scale="normal")
        ax.text(0.01, 0.04, f"measured / predicted σ = {overall:.2f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="0.7"))  # fmt: skip
        ax.set_ylim(-5e3 * np.nanmedian(predicted), 5e3 * np.nanmedian(predicted))
        ax.set_ylabel("Re V at phase centre [mJy]")
        ax.set_title(f"{label}: field {c.field}, spw {c.spw} ({pick.size} of {chan.size} visibilities shown)")
        ax.legend(fontsize=8, loc="upper right", markerscale=20)
        ratios.append(_ratio_line(axes[-1], ghz, z, label))
    if len(pooled):
        ratios.append(
            _ratio_line(axes[-1], pooled[0].freq / 1e9, _whitened(pooled), f"all {len(pooled)} fields", c="k")
        )
    every = np.concatenate(ratios)
    lo, hi = np.nanmin(every), np.nanmax(every)
    axes[-1].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))  # edge channels' bands would set the scale
    axes[-1].axhline(1, c="0.4", lw=0.8)
    axes[-1].set(xlabel="channel frequency [GHz]", ylabel="measured / predicted σ")
    axes[-1].legend(fontsize=8)
    return _save(fig, plots_dir, name)


#: Gallery frame of a line that passes only the likelihood ratio, or only the fidelity; both: the plain frame.
FRAMES = {"likelihood": "k", "fidelity": "0.8"}


def passes(cat) -> np.ndarray:
    """Which reliability test each row passes: ``both``, ``likelihood``, ``fidelity`` or ``none``."""
    from .catalogue import passes as tests

    columns = ("likelihood", "likelihood_lower_limit", "fidelity")
    if not set(columns) <= set(cat.colnames) or "k" not in cat.meta:
        return np.full(len(cat), "both")
    lr, fid = tests(*(cat[c] for c in columns), cat.meta["k"], cat.meta["fidelity_min"])
    return np.where(lr & fid, "both", np.where(lr, "likelihood", np.where(fid, "fidelity", "none")))


def _frame(ax, passed):
    """A thick frame around a panel whose line passes only one test (:data:`FRAMES`)."""
    if passed in FRAMES:
        for spine in ax.spines.values():
            spine.set_edgecolor(FRAMES[passed])
            spine.set_linewidth(3)


def _frame_note(fig, passed):
    if any(p in FRAMES for p in passed):
        fig.suptitle("thick black frame: likelihood ratio only; thick grey frame: fidelity only", fontsize=9)


def _panels(n, height, ncol=5):
    ncol = min(ncol, max(n, 1))
    nrow = max(1, int(np.ceil(n / ncol)))
    fig, axes = _subplots(nrow, ncol, figsize=(3.4 * ncol, height * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


@_styled
def line_maps_check(maps, cat, plots_dir=None, name="line_maps"):
    """Moment-0 map of every fitted line (:func:`luv_finder.imaging.line_maps`), the fitted position marked."""
    rows = cat[np.isfinite(cat["fit_dra"])]
    passed = passes(rows)
    fig, axes = _panels(len(rows), 2.9)
    for ax, row, p, (dra, ddec, m0) in zip(axes, rows, passed, maps, strict=False):
        _frame(ax, p)
        half = np.diff(dra[:2])[0] / 2
        extent = (dra[0] - half, dra[-1] + half, ddec[0] - half, ddec[-1] + half)
        # the PB correction amplifies the noise at the coverage edge: scale by the bulk, not the edge
        lo, hi = np.nanpercentile(m0, [1, 99.5]) if np.isfinite(m0).any() else (None, None)
        im = ax.imshow(m0.T, origin="lower", extent=extent, cmap="magma", vmin=lo, vmax=hi)
        ax.plot(row["fit_dra"], row["fit_ddec"], "+", c="C0", ms=12, mew=1.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jy/beam km/s")
        _sky_axes(ax)
        ax.set_title(f"#{row['id']}  {row['fit_freq_ghz']:.3f} GHz", fontsize=10)
    _frame_note(fig, passed)
    return _save(fig, plots_dir, name)


@_styled
def line_spectra_check(cat, span=3.0, plots_dir=None, name="line_spectra"):
    """Each fitted line's spectrum and best fit within ``span`` FWHM of its centre, both over the fitted peak."""
    keep = np.array([str(i) in cat.meta.get("spectra", {}) for i in cat["id"]], dtype=bool)
    rows, passed = [r for r, k in zip(cat, keep, strict=True) if k], passes(cat)[keep]
    fig, axes = _panels(len(rows), 3.2)
    for ax, row, p in zip(axes, rows, passed, strict=False):
        _frame(ax, p)
        sp = {k: np.asarray(v) for k, v in cat.meta["spectra"][str(row["id"])].items()}
        near = np.abs(sp["freq_ghz"] / row["fit_freq_ghz"] - 1) * C_KMS <= span * row["fit_width"]
        f, peak = sp["freq_ghz"][near], row["fit_peak"]
        ax.fill_between(
            f,
            (sp["flux"] - sp["error"])[near] / peak,
            (sp["flux"] + sp["error"])[near] / peak,
            step="mid",
            color="0.8",
            lw=0,
        )
        ax.step(f, sp["flux"][near] / peak, where="mid", c="k", lw=1)
        ax.plot(f, sp["model"][near] / peak, c="C3", lw=1.5)
        ax.axhline(0, c="0.5", lw=0.5)
        ax.set_title(f"#{row['id']}  S/N {row['snr']:.1f}", fontsize=10)
        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("flux / fitted peak")
    _frame_note(fig, passed)
    return _save(fig, plots_dir, name)


def _marks(marks):
    """``{label: (dra, ddec)}`` from a dict or a catalogue-like table (rows marked ``#id`` at ``dra``, ``ddec``)."""
    if marks is not None and hasattr(marks, "colnames"):
        return {f"#{r['id']}": (float(r["dra"]), float(r["ddec"])) for r in marks}
    return marks


@_styled
def peak_snr_map(maps, marks=None, plots_dir=None, name="peak_snr"):
    """Peak S/N on the sky from saved maps ``(dra, ddec, peak)`` (one or several), their maximum.

    The units of one band are searched on lattices of different steps (half each one's resolution), so
    every map is put onto the finest by nearest neighbour; positions no map covers stay blank. Drawn
    interpolated between the lattice points (Gouraud shading; display only). ``marks``
    maps a label to a (dra, ddec), or is a catalogue whose rows are marked ``#id`` at ``dra``, ``ddec``.
    """
    if isinstance(maps[0], np.ndarray) and maps[0].ndim == 1:
        maps = [maps]
    maps = [(np.asarray(x), np.asarray(y), np.asarray(z)) for x, y, z in maps]
    step = min(min(np.median(np.diff(x)), np.median(np.diff(y))) for x, y, _ in maps)
    xs = np.arange(min(x[0] for x, _, _ in maps), max(x[-1] for x, _, _ in maps) + step / 2, step)
    ys = np.arange(min(y[0] for _, y, _ in maps), max(y[-1] for _, y, _ in maps) + step / 2, step)
    peak = np.full((xs.size, ys.size), np.nan)
    for x, y, z in maps:
        ix, iy = (np.rint((a - b[0]) / np.median(np.diff(b))).astype(int) for a, b in ((xs, x), (ys, y)))
        inside = np.outer((ix >= 0) & (ix < x.size), (iy >= 0) & (iy < y.size))
        values = z[np.clip(ix, 0, x.size - 1)][:, np.clip(iy, 0, y.size - 1)]
        peak = np.fmax(peak, np.where(inside, values, np.nan))
    fig, ax = _subplots(figsize=(6.4, 5.4))
    mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(peak.T), shading="gouraud", vmin=0, cmap="magma", rasterized=True)
    fig.colorbar(mesh, ax=ax, label="peak S/N")
    _mark(ax, _marks(marks))
    _sky_axes(ax)
    ax.set_title(f"Peak S/N over templates and channels: max {_nanmax(peak):.1f}")
    return _save(fig, plots_dir, name)


def _per_group(x, n, ndim):
    """``x`` per group: a sequence of ``n``, or one ``ndim``-dimensional array for all (``ndim + 1``: stacked)."""
    if isinstance(x, np.ndarray):
        return [x] * n if x.ndim == ndim else list(x)
    return list(x)


@_styled
def filter_responses(
    freqs, snr, widths, labels, jk=None, at=None, passed=None, span=3000.0, plots_dir=None, name="filter_responses"
):
    """The matched-filter response of every group: S/N against frequency for each template, one panel each.

    ``freqs[g]`` (Hz), ``widths[g]`` (km/s) and ``snr[g]``, ``jk[g]`` (templates x frequencies) per group, as
    saved by a run (groups of different units have different frequencies); one shared array is accepted for
    each too. ``at[g]`` (Hz) is the group's own frequency, marked: another line at the same position can be
    brighter (default: the peak). The best template there is drawn thick, the jackknife's response to it in
    grey, the group's S/N dotted; each panel spans ``span`` km/s around it (None: everything). ``passed[g]``
    (see :func:`passes`) frames a group that passes only one reliability test.
    """
    n = len(labels)
    freqs, widths = _per_group(freqs, n, 1), _per_group(widths, n, 1)
    snr = _per_group(snr, n, 2)
    jk = [None] * n if jk is None else _per_group(jk, n, 2)
    passed = ["both"] * n if passed is None else list(passed)
    fig, axes = _panels(n, 2.9)
    for g, (ax, label) in enumerate(zip(axes, labels, strict=False)):
        _frame(ax, passed[g])
        f, s, w = np.asarray(freqs[g]) / 1e9, np.asarray(snr[g]), np.asarray(widths[g])
        if not np.isfinite(s).any():
            ax.set_title(f"{label}: no response", fontsize=9)
            continue
        if at is None:
            t, k = np.unravel_index(np.nanargmax(s), s.shape)
        else:
            k = int(np.argmin(np.abs(f - at[g] / 1e9)))
            t = int(np.nanargmax(s[:, k]))
            ax.axvline(f[k], c="C3", lw=0.6, alpha=0.6)
        near = np.ones(f.size, bool) if span is None else np.abs(f / f[k] - 1) * C_KMS <= span
        ax.axhline(0, c="0.5", lw=0.5)
        if jk[g] is not None:
            ax.plot(f[near], np.asarray(jk[g])[t][near], c="C7", lw=0.8, label="jackknife")
        for i in range(len(w)):
            ax.plot(f[near], s[i][near], c=f"C{i}", lw=1.6 if i == t else 0.8, label=f"{w[i]:g} km/s")
        ax.axhline(s[t, k], c="C3", ls=":", lw=0.8)
        ax.set_title(f"{label}  S/N {s[t, k]:.1f}", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("S/N")
    if n:
        axes[0].legend(fontsize=6)
    _frame_note(fig, passed)
    return _save(fig, plots_dir, name)
