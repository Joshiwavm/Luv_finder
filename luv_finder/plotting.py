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
