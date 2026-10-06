"""Primary beam: does ``simobserve`` attenuate the sky like :func:`luv_finder.utils.primary_beam`?

``configs/mocks/pb_offsets.yaml`` puts a point-source line at 0, 15, 25 and 35 arcsec from the phase centre,
each at its own frequency. A matched filter with the source's own template, run on the noiseless visibilities
without continuum subtraction (the mock has none), returns the best-fit peak flux density, so its ratio to
the injected peak is the beam simobserve applied.

CASA models the 12 m beam as an Airy pattern of a 10.7 m aperture with a 0.75 m blockage. Measured here, its
FWHM is 1.165 lambda / D (least squares over the four sources: 1.164), against the 1.13 lambda / D the ALMA
Technical Handbook gives for the beam measured on the real antennas, which ``luv_finder.utils`` uses by default.
The two differ because CASA's model is an idealised aperture, not the measured antenna, so the default beam
predicts a simobserve mock up to 4.7 per cent too low at 35 arcsec (+0.4, +1.8, +2.8 and +4.7 per cent at 0, 15,
25 and 35 arcsec): hence the 6 per cent tolerance on the default beam. The comparison is printed
(``pytest -s``).

The mock is exported to ``output/npz/pb_offsets{,_noiseless}.npz`` and rebuilt whenever the YAML is newer.
"""

import dataclasses
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from scipy.optimize import least_squares

from luv_finder import DataHandler, Gaussian, MatchedFilter, MockObservation, Model
from luv_finder.utils import FWHM_FACTOR, primary_beam

REPO = Path(__file__).parents[1]
CONFIG = REPO / "configs" / "mocks" / "pb_offsets.yaml"
NPZ_DIR = REPO / "output" / "npz"


def _observe(mock: MockObservation) -> tuple[Path, Path]:
    """Noisy and noiseless NPZ of the preset, simulated with CASA if missing or older than the YAML.

    Leaves the injected cube in ``mock.cube`` either way.
    """
    noisy, clean = NPZ_DIR / "pb_offsets.npz", NPZ_DIR / "pb_offsets_noiseless.npz"
    if all(p.exists() and p.stat().st_mtime > CONFIG.stat().st_mtime for p in (noisy, clean)):
        mock.create_cube()
    else:
        pytest.importorskip("casatasks", exc_type=ImportError)
        mock.run_all()
        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        DataHandler(mock.ms_noisy).to_npz(str(noisy))
        DataHandler(mock.ms_noiseless).to_npz(str(clean))
    return noisy, clean


def _expected(noisy_path: Path, clean_path: Path) -> DataHandler:
    """The noiseless signal with the noise model of the noisy data."""
    noisy = DataHandler.from_npz(str(noisy_path))
    (signal,) = DataHandler.from_npz(str(clean_path)).chunks
    (chunk,) = noisy.chunks
    assert np.array_equal(signal.time, chunk.time)
    return DataHandler(chunks=[dataclasses.replace(chunk, X=chunk.w * signal.vis)], metadata=noisy.metadata)


def _fitted_flux(data: DataHandler, source: dict, sigma: float) -> np.ndarray:
    """Best-fit peak flux density (Jy) per channel of a point source's template at its position."""
    comp = Gaussian()
    comp.grid = {
        "dra": source["position"][0],
        "ddec": source["position"][1],
        "bmin": sigma,
        "bmaj": sigma,
        "width": source["line"]["width"],
    }
    mod = Model()
    mod.addcomponent(comp)
    mf = MatchedFilter(data, mod, continuum_order=None)
    mf.run()
    return mf.result.flux[0, 0, 0]


@pytest.mark.casa
def test_simobserve_beam_follows_airy():
    """The fitted-to-injected peak ratio is 1 at the phase centre and follows the default Airy beam outside it."""
    mock = MockObservation.from_yaml(CONFIG)
    data = _expected(*_observe(mock))
    sigma = u.Quantity(mock.cell).to(u.arcsec).value
    injected = mock.cube.sum(axis=(1, 2))
    assert len(data.freqs) == len(injected)

    offsets, freqs, measured = [], [], []
    for source in mock.sources:
        chan = int(np.argmin(abs(data.freqs - source["line"]["mean"] * 1e9)))
        assert data.freqs[chan] == pytest.approx(source["line"]["mean"] * 1e9, abs=1e3)
        offsets.append(np.hypot(*source["position"]))
        freqs.append(data.freqs[chan])
        measured.append(_fitted_flux(data, source, sigma)[chan] / injected[chan])
    offsets, freqs, measured = map(np.array, (offsets, freqs, measured))

    dish = data.metadata.dish_diameter
    beam = primary_beam(offsets, freqs, dish)
    best = least_squares(lambda k: primary_beam(offsets, freqs, dish, fwhm_factor=k[0]) - measured, [FWHM_FACTOR]).x[0]
    print(f"\nbest-fit FWHM factor {best:.4f} lambda/D (package default {FWHM_FACTOR})")
    print("offset/arcsec  freq/GHz  measured  PB(default)  diff/%")
    for off, freq, meas, pb in zip(offsets, freqs, measured, beam, strict=True):
        print(f"{off:13.1f}  {freq / 1e9:8.3f}  {meas:8.4f}  {pb:11.4f}  {100 * (meas / pb - 1):+6.2f}")

    assert measured[0] == pytest.approx(1.0, rel=0.01)
    assert measured == pytest.approx(beam, rel=0.06)
