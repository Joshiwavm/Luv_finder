# Roadmap

Current state: grid-search matched filter on simulated single-pointing data.
Nested sampling was removed (grid search is sufficient for now).

1. **Merge with [alma-data-prep](https://github.com/Joshiwavm/alma-data-prep)**
   The end goal is one repository; the mechanism is undecided. Interim: install
   `alma-data-prep` for organising/concatenating real archives and read its UV
   exports through `DataHandler.from_npz`.

2. **JAX acceleration**
   Port `Gaussian._uvgauss_1D2D`, `MatchedFilter.delay_transform` and the grid loop
   to `jax.numpy` with `vmap` over grid points. CPU on macOS, CUDA on the cluster.
   All science code downstream of `data.py` is already numpy-only, so CASA and JAX
   never need to meet.

3. **Tests from simulated cases**
   Add a bright + faint two-line fixture. Bright-line subtraction: detect with the
   moment-8 machinery in `alma_data_prep.export_cube.ExportCube`, subtract the
   best-fit UV model, re-run the finder on the residual.

4. **Catalogue and detection inference**
   `luv-find --catalog`: peaks above threshold in the response cube. Null
   distribution from jackknife responses gives the false-positive rate per
   threshold.

5. **Primary-beam attenuation and mosaics**
   Open problem: pointings overlapping the same sky position have different PB
   attenuation. Candidate approach: PB factor per pointing in the UV model
   (`Metadata.primarybeamsize` -> Gaussian PB), joint response summed over
   pointings with PB^2 weighting. Needs multi-field support in `DataHandler`.

6. **Other exploration methods** beyond grid search (TBD).
