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
| Spectral windows | 4 | 4 |
| Coverage | 37.7-45.6 GHz | 84.0-100.0 GHz |
| Channels per window | 128, 128, 128, **960** | 128 x 4 |
| Velocity resolution | 105-121 km/s, **13.7 km/s** | 47-55 km/s |
| Antennas | 50 | 51 |
| Rows | 1.75 M | 5.50 M |
| Expected lines | ~4 | 11 |

Lines sit at different sky positions in both bands, so this is a genuine blind
search rather than a targeted extraction. Both sets were produced by
`alma-data-prep`, which makes its export path the natural place to put any
chunking this package needs, and is a concrete step toward merging the two.

### 0. Handle the data volume

Neither dataset fits the current in-memory representation.

| | Band 1 | Band 3 |
|---|---|---|
| On disk | 12 GB | 20 GB |
| Visibility-channels | 588 M | 704 M |
| As `uvdata` today | 37.6 GB | 45.0 GB |
| Largest single pointing | 37.6 GB | 6.4 GB |

`DataHandler` keeps eight float64 arrays, 64 bytes per visibility-channel, which
is 2-3x the size of the measurement set itself. Two changes fix this, and both
are the same refactor that the analytic work below needs:

- **Store factored, not expanded.** Only `UVreals` and `UVimags` are genuinely
  per visibility-channel. `uwaves` and `vwaves` are the outer product of `u, v`
  per row with frequency per channel; `uvfreqs` is per channel; `uvwghts` and
  `uvtimes` are per row; `uvdists` is derivable from `uwaves, vwaves`. Keeping
  the factors and forming products on demand takes 64 bytes per
  visibility-channel down to 16, and to 8 in float32 for the data arrays, which
  is smaller than the measurement set. A Band 3 pointing becomes under 1 GB.
- **Chunk by (field, spw).** `load_data` currently concatenates every field and
  window into one flat array, then reshapes to `(n_freq, n_vis)`. That reshape
  assumes a single rectangular channel grid and is simply invalid for Band 1,
  which mixes 128- and 960-channel windows, and it silently merges the seven
  Band 3 pointings. Process one (field, spw) at a time and combine at the
  response level, which is also what the mosaic step needs.

This is a prerequisite for every step below.

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

### JAX and jit
Measured on the test fixture, 256 positions, CPU, float64 throughout. Results
agree with the current path to 5e-9 relative error.

| Path | Time | Gain |
|---|---|---|
| Current numpy | 1510 ms | — |
| Analytic, kernel hoisted out of the position loop | 166 ms | 9.1x |
| `vmap` over positions, no jit | 95 ms | |
| `vmap` + `jit` | 11 ms | 8.5x on top |

Two things follow. First, jit is worth roughly as much as the analytic
restructuring, so both are worth doing. Second, the restructuring is a
precondition rather than an alternative: `_grid_point_response` currently
deep-copies a `SimpleNamespace` and mutates model attributes with `setattr`,
neither of which is traceable, so the present code cannot be jitted at all. The
analytic form is a pure function of arrays and jits directly.

Port `Gaussian._uvgauss_1D2D`, `delay_transform` and the grid loop to
`jax.numpy`. Enable `jax_enable_x64`: the weighted sums over visibilities need
it, and jackknify already did this. CPU on macOS, CUDA on the cluster.
Everything downstream of `data.py` is numpy-only, so CASA and JAX never need to
share an environment.

Two practical constraints, both measured:

- **Recompilation on shape change.** Going from 256 to 300 positions retriggers
  compilation, 73 ms against 10 ms cached. Keep the batch shape fixed and pad
  the final chunk rather than letting the grid size vary.
- **Pick the batching primitive deliberately.** JAX does not parallelise a
  Python loop; a loop inside `jit` is unrolled at trace time. The choice is
  explicit, and it decides the memory profile:

  | 16384 positions | Time | Memory growth |
  |---|---|---|
  | `vmap` | 1250 ms | 8.6 GB |
  | `lax.map` | 1860 ms | ~0 GB |

  `vmap` materialises the `n_pos x n_freq x n_vis` intermediate; XLA does not
  fuse it away. `lax.map` sequences the batch axis and holds memory flat at
  every size tested, for about 1.5x the time. The sensible default is `lax.map`
  over chunks with `vmap` inside each chunk: memory bounded by the chunk, speed
  close to `vmap`. The NUFFT remains the real fix, since it removes the position
  axis from the per-visibility work altogether.

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
