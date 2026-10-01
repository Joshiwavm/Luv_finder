"""Diagnostic figures. Only run with --plots; they are for the eye, not for CI.

Assertions here are deliberately weak: the point is to produce something to look
at. The numerical checks live in the other test modules.
"""

import numpy as np

from luv_finder import Gaussian, MatchedFilter, Model
from luv_finder.plotting import response_check, spectrum_check


def test_spectrum_per_source(data, truth, plots):
    """Re/Im visibility spectra: data, jackknife and model, centred and shifted."""
    res = data.metadata.minresolution()
    for i, src in enumerate(truth["sources"]):
        dra, ddec = src["position_model"]
        model = Gaussian(
            dra=dra,
            ddec=ddec,
            total_flux=src["line"]["flux"],
            bmin=res / 10,
            bmaj=res / 10,
            nu_center=src["line"]["mean"] * 1e9,
            width=src["line"]["width"],
        )
        path = spectrum_check(
            data,
            dra,
            ddec,
            model=model,
            plots_dir=str(plots),
            name=f"spectrum_src{i}",
            line_ghz=src["line"]["mean"],
        )
        assert path.endswith(".png")


def test_response_figure(data, truth, plots):
    """Matched-filter S/N spectrum with the jackknife overlaid."""
    positions = np.array([s["position_model"] for s in truth["sources"]])
    g = Gaussian()
    g.grid = {
        "dra": positions[:, 0],
        "ddec": positions[:, 1],
        "bmin": data.metadata.minresolution() / 10,
        "bmaj": data.metadata.minresolution() / 10,
        "width": 300.0,
    }
    mod = Model()
    mod.addcomponent(g)
    mf = MatchedFilter(data, mod)
    mf.run(pool=1, jackknife=True)
    path = response_check(
        mf, plots_dir=str(plots), name="filter_response", line_ghz=truth["sources"][0]["line"]["mean"]
    )
    assert path.endswith(".png")
