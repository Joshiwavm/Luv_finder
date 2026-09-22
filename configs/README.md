# Configuration presets

| Directory | Contents |
|---|---|
| `alma/` | CASA antenna configuration files and the pointing file, used by `simobserve` |
| `mocks/` | `MockObservation` presets, run with `luv-mock <file>` |
| `grids/` | Matched-filter search grids, passed as `luv-find --grid <file>` |

## Pairing

Every mock preset has a grid preset of the same name. The exception is
`known_sources.yaml`, which serves all three `line*_line*` mocks because they
share the same two source positions and line frequencies and differ only in SNR.

| Mock | Grid | Array | Volume |
|---|---|---|---|
| `smoke.yaml` | `smoke.yaml` | ACA, 10 antennas | ~7 MB, ~10 s |
| `line13_line9.yaml` | `known_sources.yaml` | 12 m, 43 antennas | ~180 MB |
| `line10_line10.yaml` | `known_sources.yaml` | 12 m, 43 antennas | ~130 MB |
| `line5_line10.yaml` | `known_sources.yaml` | 12 m, 43 antennas | ~125 MB |
| any | `default.yaml` | — | blind search, grid derived from the data |

## The dra sign flip

`position: [dra, ddec]` in a mock preset is an offset in arcsec in the **image**
convention used by the cube builder. The UV model in `luv_finder.model.Gaussian`
uses the opposite sign for dra:

    dra_model = -dra_mock        ddec_model = +ddec_mock

A source injected at `[-4.25, 23.5]` is therefore recovered at `dra = +4.25,
ddec = +23.5`. Grid presets are written in model convention. This is a known
wart rather than a deliberate feature; `tests/test_mock.py` asserts the relation
so it cannot change silently.

## Naming

`MockObservation` derives the project name from the source list: one `line<snr>`
token per line source, joined by `_`. So `line13_line9` means the first source
has SNR 13 and the second SNR 9. Renaming a preset file does not rename its
products, but changing an SNR does. Set `fits_filename` to pin the name instead,
as `smoke.yaml` does.

## Test configs

They live here, not under `tests/`. `tests/test_mock.py` loads `mocks/smoke.yaml`
and `grids/smoke.yaml` rather than hardcoding parameters, so the presets and the
test cannot drift apart, and anything the test does can be reproduced by hand
with the two `luv-` commands.
