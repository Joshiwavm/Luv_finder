# Roadmap

Current state: grid-search matched filter, in S/N units, on simulated
single-pointing data. Nested sampling was removed; grid search is sufficient for
now.

## Real data: SPT-CL J0459-4947

Two 12 m datasets on the same source, to be worked through in the order below.
Each step is a gate: do not start the next one until the previous is understood.

| | Band 1 | Band 3 |
|---|---|---|
| Pointings | 1 | 7 (mosaic) |
| Spectral windows | 5, 38.7-44.9 GHz | 4, 85.0-99.0 GHz |
| Antennas | 50 | 51 |
| Expected lines | ~4 | 11 |

Lines sit at different sky positions in both bands, so this is a genuine blind
search rather than a targeted extraction.

**Blocked**: the measurement sets currently in `data/` are continuum-averaged,
one channel per spectral window with widths of 6000-15000 km/s. There is no
spectral axis to search. The spectral versions are being downloaded.

### 1. Decide whether continuum subtraction is needed
Run the finder on a single pointing with and without `uvcontsub`. Continuum
sources bias the matched filter because the template integrates a smooth
spectral component as if it were line flux. Compare recovered line lists and
S/N. If subtraction is needed, decide whether it belongs upstream in
`alma-data-prep` or as a step in this package. Settle the fitting order and
which channels are masked as line-contaminated.

### 2. Detection inference on one pointing
Single Band 3 pointing. Turn the response cube into a catalogue: peaks above
threshold with position, frequency, width and S/N. Calibrate the false-positive
rate from the jackknife response distribution over the same grid, so a threshold
maps to an expected number of spurious detections. Check the recovered count
against the 11 expected lines.

### 3. Joint Band 1 + Band 3 identification
One pointing per band. A line in each band at the same sky position is a
redshift confirmation, since the two bands sample different transitions of the
same ladder. Needs a co-spatial matching step, with a positional tolerance set
by the beam of the coarser band, and a way to express joint significance from
two independent responses. This is where the catalogue format has to become
cross-band.

### 4. All seven Band 3 pointings
Mosaic. The open problem: pointings overlapping the same sky position have
different primary-beam attenuation, so their visibilities cannot simply be
concatenated. Candidate approach is a per-pointing PB factor in the UV model
(`Metadata.primarybeamsize` gives the scale) and a joint response summed over
pointings with PB-squared weighting. Requires multi-field support in
`DataHandler`, which currently flattens all fields into one array.

## Performance

### Analytic evaluation instead of per-grid-point exponentials
Measured on the test fixture: 81% of the time per grid point is the model
evaluation plus the two phase shifts, all of them complex exponentials over
every visibility. Because the model is a Gaussian, most of that work is either
redundant or shareable.

- **The model phase cancels.** The model is generated at (dra, ddec) and then
  phase-shifted by the same offset, so the kernel is the phase-free envelope
  `A(u,v) * S(nu)`. Verified: the imaginary part after shifting is 1e-16 of the
  real part, and the real part equals the envelope computed directly. One
  complex exponential per grid point is pure waste.
- **The spatial taper is position-independent.** `A(u,v; bmin, bmaj)` does not
  depend on dra or ddec, so it can be computed once per source size and reused
  across the whole position grid instead of recomputed for every combination.
- **The position search is a Fourier transform.** The weighted channel mean of
  the phase-shifted data, as a function of (dra, ddec), is a non-uniform Fourier
  transform of the weighted visibilities. For a regular position grid this is
  one NUFFT per channel rather than one exponential per position, turning
  `O(N_pos * N_vis)` into roughly `O(N_vis log N + N_pos)`. `jax-finufft` is
  already in the `jax` extra and was used this way in the jackknify package.
- **The template transform is closed-form.** The Fourier transform of a Gaussian
  is a Gaussian, so `fft(kernel)` in `delay_transform` can be written down
  analytically rather than computed.

### JAX
Port `Gaussian._uvgauss_1D2D`, `delay_transform` and the grid loop to
`jax.numpy` with `vmap` over grid points. Do the analytic restructuring first:
it removes most of the work rather than moving it to a faster device, and what
remains maps cleanly onto `vmap` and NUFFT primitives. CPU on macOS, CUDA on the
cluster. Everything downstream of `data.py` is already numpy-only, so CASA and
JAX never need to share an environment.

## Longer term

- **Merge with [alma-data-prep](https://github.com/Joshiwavm/alma-data-prep).**
  The end goal is one repository; the mechanism is undecided. Interim: use it for
  organising, concatenating and exporting real archives, and read its UV exports
  through `DataHandler.from_npz`.
- **Bright-line subtraction.** Detect with the moment-8 machinery in
  `alma_data_prep.export_cube.ExportCube`, subtract the best-fit UV model, re-run
  the finder on the residual. Needs a two-line fixture, bright plus faint.
- **The `dra` sign flip** between image and model conventions is documented and
  asserted but not fixed. Fixing it means regenerating the committed fixture.
- **Other exploration methods** beyond grid search, once the grid search is
  understood on real data.
