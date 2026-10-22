# Changelog

Notable changes, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Matched filter in JAX (float64): every position of a window is one complex matrix
  product per channel, with the phase factors advanced by recurrence.
- A polynomial continuum fitted jointly with every line template (`continuum_order`,
  default 1), refitted at each position without the channels of its other lines at
  S/N >= 5, in two passes, a centre's own line exempt.
- Correlated channel noise: the correlation is measured on the jackknife per spectral
  window, and every template is normalised by its exact variance.
- Analytic ALMA primary beam, per-pointing primary-beam correction and inverse-variance
  mosaic combination (`search_pointings`).
- Line catalogue with a likelihood ratio and a fidelity against the jackknife, and a
  least-squares line fit in the visibilities, run by `luv-run` (`Run`): one YAML in,
  every product in one directory.
- Spectral windows that overlap or whose gap is narrower than the widest template are
  searched and fitted jointly; a line across a window edge is one line.
- Frequency chunking: `max_channels` cuts a unit into chunks that overlap by a block and
  give the unchopped catalogue.
- NUFFT dirty cubes, moment-8, continuum and moment-0 maps for inspection.
- A position angle for the Gaussian template.
- `CITATION.cff`, `CONTRIBUTING.md` and this changelog.

### Changed

- Every source size is a FWHM, like the line width: grid, catalogue and fit `bmaj`,
  `bmin`, and mock `axis_maj`, `axis_min`.
- `MatchedFilter(data, grid)` takes the grid dict; unknown grid and run keys raise.
- `luv-find` is a quick look: every field (or `--field N`) PB-corrected and combined.
- Weights are corrected on reading per scan and channel, from pairs within a field first.
- The primary beam uses the Airy series near a pointing centre (JAX's Bessel derivative
  is wrong there).
- The line fit runs one compiled step per pointing (10x faster); the masked continuum
  refit works one position at a time; the search keeps no copies of its results.

- Visibilities are stored per (field, spectral window) as `X = w V` with flags; Stokes I
  from `WEIGHT` and `FLAG`.
- Positions are arcsec east and north in one sky frame per dataset, mocks included.
- One weighting: each visibility is weighted by the trial source's own amplitude.
- Python 3.12 is required.
- The documentation is split into a quickstart, a one-page method, and guides.

### Removed

- `luv-export` (a run exports), `luv-find --catalogue/--fit`, the `Model` container,
  `utils.getstatwtweights`, the run's explicit `units` key.

### Fixed

- `fit_line(channel_noise="measure")` measured the channel correlation on the data instead
  of the jackknife.
- A mock no longer deletes the `*.log` files of its working directory.

- The jackknife pairs integrations by baseline and gives each pair the weight
  `4 w_a w_b / (w_a + w_b)`.

## [0.2.0] - 2026-09-22 (untagged)

- First packaged version: the `luv_finder` package, the `luv-mock`, `luv-export` and
  `luv-find` command-line tools, a CASA-free test suite, Sphinx documentation and CI.
