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
    ax.set_title(f"Peak S/N {best.max():.1f} at {freqs[np.argmax(best)]:.3f} GHz")
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
    ``marks`` maps a label to a (dra, ddec) position to circle, e.g. an injected source.
    """
    dra, ddec = np.asarray(mf.axes["dra"]), np.asarray(mf.axes["ddec"])
    panels = {"data": mf.response}
    if mf.response_jackknife is not None:
        panels["jackknife"] = mf.response_jackknife
    peaks = {k: r.reshape(len(dra), len(ddec), -1).max(axis=-1) for k, r in panels.items()}
    vmax = max(p.max() for p in peaks.values())
    fig, axes = plt.subplots(1, len(peaks), figsize=(5.2 * len(peaks), 4.6), squeeze=False)
    for ax, (label, peak) in zip(axes[0], peaks.items(), strict=True):
        mesh = ax.pcolormesh(dra, ddec, peak.T, shading="nearest", vmin=0, vmax=vmax, cmap="magma")
        fig.colorbar(mesh, ax=ax, label="peak S/N")
        _mark(ax, marks)
        _sky_axes(ax)
        ax.set_title(f"{label}: max {peak.max():.1f}")
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
    """
    from scipy.stats import norm

    panels = {"data": mf.response, "jackknife": mf.response_jackknife}
    panels = {k: v.ravel() for k, v in panels.items() if v is not None}
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
    Contours are at -3 (dashed) and 3, 5, 10, 20, 50 times the continuum noise ``sigma``.
    """
    dra, ddec = np.asarray(dra), np.asarray(ddec)
    levels = np.array([3, 5, 10, 20, 50]) * sigma
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = (
        (axes[0], moment8, "moment-8 (line S/N)", "S/N", "magma"),
        (axes[1], 1e3 * continuum, f"continuum, noise {1e6 * sigma:.1f} uJy/beam", "mJy/beam", "viridis"),
        (axes[2], moment8, "moment-8 + continuum contours", "S/N", "magma"),
    )
    for ax, image, title, unit, cmap in panels:
        mesh = ax.pcolormesh(dra, ddec, image.T, shading="nearest", cmap=cmap)
        fig.colorbar(mesh, ax=ax, label=unit)
        ax.set_title(title)
        _mark(ax, marks)
        _sky_axes(ax)
    axes[2].contour(dra, ddec, continuum.T, levels=levels, colors="w", linewidths=0.8)
    axes[2].contour(dra, ddec, continuum.T, levels=[-3 * sigma], colors="w", linewidths=0.8, linestyles="--")
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
