# Presets

`mocks/` are `MockObservation` inputs (`luv-mock <file>`); `grids/` are search
grids (`luv-find --grid <file>`). Each mock has a grid of the same name.

| Preset | Array | Volume |
|---|---|---|
| `smoke` | ACA, 10 antennas | ~7 MB, ~20 s — used by `tests/test_mock.py` |
| `line13_line9` | 12 m, 43 antennas | ~180 MB |
| `size_point`, `size_beam`, `size_3beam` | 12 m | one source of 0, 1 and 3 beam FWHM, used by `tests/test_source_size.py` |
| `default` grid | — | blind search, ranges derived from the data |

Antenna configurations are not vendored. `alma_config` names a file that CASA
ships, such as `alma.cycle13.3.cfg`, resolved from its data directory.

**`dra` is sign-flipped between the two conventions.** Mock `position` is in image
convention; the UV model uses `dra_model = -dra_mock`, `ddec_model = +ddec_mock`.
Grids are written in model convention. `tests/test_mock.py` asserts the relation.

`total_flux` is not a grid key: it cancels in the matched-filter normalisation, so
the response is already in S/N units and varying the flux only duplicates points.
