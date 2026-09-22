import pytest

pytest.importorskip("casatasks")

from luv_finder import DataHandler, MockObservation  # noqa: E402


@pytest.mark.casa
def test_tiny_mock_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    import shutil
    from pathlib import Path

    shutil.copytree(Path(__file__).parents[1] / "configs" / "alma", tmp_path / "configs" / "alma")

    mock = MockObservation(
        cube_shape=(8, 32, 32),
        cell="0.5arcsec",
        integration_time="1min",
        sources=[{"position": (0.0, 0.0), "line": {"width": 300, "mean": 40.0, "snr": 20}}],
        output_folder=str(tmp_path / "out"),
        support_folder=str(tmp_path / "support"),
    )
    dest = mock.run_all(plots_dir=str(tmp_path / "plots"))
    data = DataHandler(mock.ms_noisy)
    assert data.n_freqs(data.uvdata) == 8
    assert data.uvdata.UVreals.size > 0
    assert (tmp_path / "out").exists() and dest.startswith(str(tmp_path))
