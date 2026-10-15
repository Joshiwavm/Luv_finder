"""The continuum polynomial: one default for search and fit, and mock continua with a spectral index."""

import numpy as np
import pytest

from luv_finder.mock import MockObservation
from luv_finder.model import CONTINUUM_ORDER
from luv_finder.run import Run


@pytest.mark.parametrize("index", [0.0, 3.5, -1.0])
def test_mock_continuum_follows_its_spectral_index(index):
    mock = MockObservation(
        cube_shape=(20, 32, 32),
        freq_center="85GHz",
        dv="55km/s",
        integration_time="10min",
        sources=[{"position": (0.0, 0.0), "continuum": {"snr": 10.0, "index": index}}],
    )
    cube = mock.create_cube()
    df, f_start, _ = mock._freq_axis()
    nu = f_start + df * np.arange(20)
    spectrum = cube[:, 16, 16]
    np.testing.assert_allclose(spectrum / spectrum[10], (nu / nu[10]) ** index, rtol=1e-12)


@pytest.mark.parametrize(("filter_cfg", "order"), [({}, CONTINUUM_ORDER), ({"continuum_order": 2}, 2),
                                                    ({"continuum_order": None}, None)])  # fmt: skip
def test_run_fits_with_the_degree_it_searched_with(tmp_path, filter_cfg, order):
    run = Run({"data": "data.npz", "output": str(tmp_path), "filter": filter_cfg})
    assert run.fit_options == {"continuum_order": order}
