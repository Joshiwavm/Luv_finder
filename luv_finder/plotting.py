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


def _weighted_spectrum(uvdata, n_freq, n_vis, attr="UVreals"):
    """Weighted mean of ``attr`` over visibilities, one value per channel."""
    return np.average(
        getattr(uvdata, attr).reshape(n_freq, n_vis),
        weights=uvdata.uvwghts.reshape(n_freq, n_vis),
        axis=1,
    )


def spectrum_check(data, dra, ddec, model_uv=None, plots_dir="plots", name="spectrum", line_ghz=None):
    """Real and imaginary visibility spectra: data, jackknife and model.

    Each is shown at the phase centre and phase-shifted onto (``dra``, ``ddec``).
    A real source is flat at the phase centre and peaks once shifted; the
    jackknife should stay consistent with zero in both.
    """
    nf, nv = data.n_freqs(data.uvdata), data.n_visbs(data.uvdata)
    freqs = data.uvdata.uvfreqs.reshape(nf, nv)[:, 0] / 1e9

    shifted = data.apply_phase_shift(dra, ddec, data.uvdata)
    jack = data.jackknife(data.uvdata)
    jack_shift = data.apply_phase_shift(dra, ddec, jack)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.5))
    for ax, part in zip(axes, ("UVreals", "UVimags"), strict=True):
        sh = part + "_shifted"
        ax.axhline(0, c="gray", ls="--", lw=0.8)
        ax.plot(
            freqs, _weighted_spectrum(data.uvdata, nf, nv, part), c="C0", lw=1, alpha=0.6, label="data, phase centre"
        )
        ax.plot(freqs, _weighted_spectrum(shifted, nf, nv, sh), c="C1", lw=1.6, label="data, shifted")
        ax.plot(freqs, _weighted_spectrum(jack_shift, nf, nv, sh), c="C7", lw=1, ls=":", label="jackknife, shifted")
        if model_uv is not None:
            ms = data.apply_phase_shift(dra, ddec, model_uv)
            ax.plot(freqs, _weighted_spectrum(ms, nf, nv, sh), c="C2", lw=1.2, alpha=0.8, label="model, shifted")
        if line_ghz is not None:
            ax.axvline(line_ghz, c="C3", ls="--", lw=0.8)
        ax.set_ylabel(f"{'Re' if part == 'UVreals' else 'Im'}(V)  [Jy]")
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
