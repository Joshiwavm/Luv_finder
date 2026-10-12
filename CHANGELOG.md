# Changelog

Notable changes, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Matched filter in JAX (float64): every position of a window is one complex matrix
  product per channel, with the phase factors advanced by recurrence.
- A polynomial continuum fitted jointly with every line template (`continuum_order`,
  default 2).
- Correlated channel noise: the correlation is measured on the jackknife per spectral
  window, and every template is normalised by its exact variance.
- Analytic ALMA primary beam, per-pointing primary-beam correction and inverse-variance
  mosaic combination (`search_pointings`).
- Line catalogue with a likelihood ratio and a fidelity against the jackknife
  (`luv-find --catalogue`).
- Least-squares line fit in the visibilities (`luv-find --fit`).
- NUFFT dirty cubes, moment-8, continuum and moment-0 maps for inspection.
- A position angle for the Gaussian template.
- `CITATION.cff`, `CONTRIBUTING.md` and this changelog.

### Changed

- Visibilities are stored per (field, spectral window) as `X = w V` with flags; Stokes I
  from `WEIGHT` and `FLAG`.
- Positions are arcsec east and north in one sky frame per dataset, mocks included.
- One weighting: each visibility is weighted by the trial source's own amplitude.
- Python 3.12 is required.
- The documentation is split into a quickstart, a one-page method, and guides.

### Fixed

- The jackknife pairs integrations by baseline and gives each pair the weight
  `4 w_a w_b / (w_a + w_b)`.

## [0.2.0] - 2026-09-22 (untagged)

- First packaged version: the `luv_finder` package, the `luv-mock`, `luv-export` and
  `luv-find` command-line tools, a CASA-free test suite, Sphinx documentation and CI.
