# Roadmap

Current state: grid-search matched filter, in S/N units, on simulated
single-pointing data. Work the three blocks below in order: the data
representation has to change before performance work is worth doing, and both
have to land before the real data is tractable.

## 1. Data volume and layout

Neither real dataset fits the current in-memory representation.

| | Band 1 | Band 3 |
|---|---|---|
| On disk | 12 GB | 20 GB |
| Visibility-channels | 588 M | 704 M |
| As `uvdata` today | 37.6 GB | 45.0 GB |

### Where the inflation comes from

`DataHandler` keeps eight float64 arrays, every one of them the full
`n_freq x n_vis` length, so 64 bytes per visibility-channel. Only two of them
carry that much information. Band 3, per array:

| Array | Stored | Distinct values | Waste |
|---|---|---|---|
| `UVreals`, `UVimags` | 5.6 GB each | 5.6 GB each | none |
| `uwaves`, `vwaves` | 5.6 GB each | 44 MB each | 5.6 GB each |
| `uvwghts`, `uvtimes` | 5.6 GB each | 44 MB each | 5.6 GB each |
| `uvdists` | 5.6 GB | derivable | 5.6 GB |
| `uvfreqs` | 5.6 GB | 1 KB | 5.6 GB |

`uvfreqs` is the clearest case: 128 distinct frequencies, written out 704 million
times. `uwaves` is `u * nu / c`, an outer product of 5.5 M baseline coordinates
with 128 frequencies, stored as if all 704 M entries were independent. Nothing is
wrong with the numbers, they are just the same values repeated.

### Fix: store the factors, whiten once

- Keep `UVreals` and `UVimags` two-dimensional. Keep `u`, `v`, `time` per row and
  frequency per channel, and form `uwaves`, `uvfreqs` and `uvdists` on demand.
- **Whiten on load.** Since everything downstream works in S/N units, store
  `d * sqrt(w)` rather than `d` and `w` separately. The weights then never need
  to be kept at full length; `sqrt(w)` per row is enough to whiten a template.
- Stay in float64. The weighted sums run over 10^8 terms and float32 does not
  have the precision for them.

Band 3 goes from 45.0 GB to 11.4 GB, which is 1.6 GB per pointing. Band 1 goes
from 37.6 GB to 9.5 GB. Both then fit comfortably.

The uv binning in section 2 attacks the same problem from the other side, by
reducing the number of visibilities rather than the bytes per visibility, and is
worth a factor of tens. If it holds up, it largely dissolves this section.

### Fix: chunk by (field, spw)

`load_data` concatenates every field and window into one flat array and then
reshapes to `(n_freq, n_vis)`. That reshape assumes a single rectangular channel
grid. It is invalid for Band 1, which mixes three 128-channel windows with one
960-channel window, and it silently merges the seven Band 3 pointings into one
block. Process one (field, spw) at a time and combine at the response level,
which is what the mosaic step needs anyway.

Both datasets came from `alma-data-prep`, so its export path is the natural place
to put this chunking, and doing it there is a concrete first step toward merging
the two repositories.

## 2. Performance

### Analytic evaluation instead of per-grid-point exponentials

81% of the time per grid point is the model evaluation plus the two phase
shifts, all complex exponentials over every visibility. Because the model is a
Gaussian, nearly all of it is redundant.

- **The model phase cancels.** The model is generated at (dra, ddec) and then
  phase-shifted by the same offset, so the kernel is the phase-free envelope
  `A(u,v) * S(nu)`. Verified: the imaginary part after shifting is 1e-16 of the
  real part. One complex exponential per grid point is pure waste.
- **The kernel is position-independent.** `A(u,v; bmin, bmaj)` does not contain
  dra or ddec, so the normalised kernel is identical at every position, verified
  to 3e-14. It needs computing once per (size, width), not once per grid point.
- **The template transform is closed-form.** The Fourier transform of a Gaussian
  is a Gaussian, so `fft(kernel)` can be written down instead of computed.

### uv binning, corrected for the primary beam

Averaging visibilities that land in the same uv cell reduces `N_vis`, which is
the dominant axis in both the memory problem of section 1 and the cost here.
Measured on Band 3, field 0, window 0: 196,508 rows at 85.0 GHz, longest
baseline 100 klambda, primary beam 68.5 arcsec.

| Cell | Field of view | Occupied cells | Reduction |
|---|---|---|---|
| 6.0 klambda | 0.5 x PB | 462 | 425x |
| 3.0 klambda | 1.0 x PB | 1,540 | 128x |
| 1.5 klambda | 2.0 x PB | 4,889 | 40x |

The coverage is sparse relative to the number of (baseline, time) samples, so
many integrations fall in the same cell. Searching the full primary beam wants
the last row. Applied across every field and window that takes Band 3 from 704 M
visibility-channels to roughly 18 M, so 45 GB becomes about 1.1 GB.

What it costs, and what has to be right:

- **The cell size is the field of view.** `cell = 1 / FoV` is the usual gridding
  criterion, and sources toward the edge of that field are attenuated by the
  transform of the cell. Measure the S/N loss against source offset on a
  simulated field before committing to a cell size; the usable search radius is
  smaller than the nominal one.
- **Bin within (field, spw), never across pointings.** Visibilities from
  different pointings carry differently PB-weighted skies, so averaging them into
  one cell mixes them. Combining pointings stays at the response level in 3.4.
- **Divide the recovered S/N by PB(theta).** The primary beam attenuates
  off-axis sources within every pointing, so a raw response understates the
  intrinsic line flux away from the phase centre.
- **Do not bin in frequency.** The lines are narrow, and Band 1's 960-channel
  window exists precisely to resolve them.
- `u` and `v` scale with frequency, so a grid fixed in wavelengths is exact at
  one frequency only. Per window the fractional bandwidth is about 2%, so
  gridding at the window centre is accurate to that level. Check it is
  acceptable rather than assuming.
- Bin weight-aware, `V_cell = sum(w V) / sum(w)` with `w_cell = sum(w)`, which
  the whitened storage of section 1 provides directly.

This overlaps with the NUFFT below, since gridding is the first step of a type-1
NUFFT, but the two are complementary. Binning is far cheaper than a transform
and shrinks every downstream array, and once the data are a few thousand points
the NUFFT cost is dominated by its FFT rather than by spreading. Bin first, then
transform.

### The position search is a Fourier transform

The response is built along two different axes, and only the first is expensive.

**Spatial.** For a trial position the data are phase-shifted and collapsed to one
number per channel:

```
d(nu; dra, ddec) = sum_j w_j Re[ V_j(nu) exp(-2i pi (u_j dra + v_j ddec)) ] / sum_j w_j
```

Read as a function of `(dra, ddec)`, that is a Fourier transform of the weighted
visibilities from the irregular `(u, v)` samples onto a regular position grid: a
type-1 NUFFT, and physically just the dirty image of that channel. Looping over
positions is therefore computing the dirty image one pixel at a time, at
`O(N_pos * N_vis)`. One NUFFT per channel produces every position at once, at
roughly `O(N_vis + N_grid log N_grid)`. It has to be per channel because
`u * nu / c` rescales with frequency, which is ordinary spectral-cube gridding.

**Spectral.** `delay_transform` slides the template along frequency. This part
does not change. It runs on an array of shape `(n_pos, n_freq)`, negligible next
to the visibilities, and vectorises trivially.

So the restructured pipeline is: one NUFFT per channel to build the weighted
dirty cube, one kernel normalisation per template shape, then the existing FFT
along frequency for every position. The two transforms are along different axes
and compose; the NUFFT replaces the phase-shift loop, not `delay_transform`.

One constraint this introduces: the position grid must be regular. The current
code accepts arbitrary `dra`/`ddec` lists, which stays useful for targeted
checks, but a blind search wants a regular grid anyway.

### JAX, jit and large volumes

Measured on the fixture, 256 positions, CPU, float64, agreeing with the current
path to 5e-9:

| Path | Time |
|---|---|
| Current numpy | 1510 ms |
| Analytic, kernel hoisted | 166 ms |
| `vmap`, no jit | 95 ms |
| `vmap` + `jit` | 11 ms |

The restructuring is also a precondition, not an alternative:
`_grid_point_response` deep-copies a `SimpleNamespace` and mutates model
attributes with `setattr`, neither of which is traceable, so the present code
cannot be jitted at all.

**Batching is an explicit choice.** JAX does not parallelise a Python loop; a
loop inside `jit` is unrolled at trace time. Measured at 16384 positions:

| | Time | Memory growth |
|---|---|---|
| `vmap` | 1250 ms | 8.6 GB |
| `lax.map` | 1860 ms | ~0 GB |

`vmap` materialises the `n_pos x n_freq x n_vis` intermediate and XLA does not
fuse it away. `lax.map` sequences the batch and holds memory flat for about 1.5x
the time.

**Fitting the real data.** Even factored, a Band 3 pointing is 1.6 GB of
visibilities, and the NUFFT output is a dirty cube of `n_pos x n_freq x 8` bytes,
which for a 512 x 512 grid over 128 channels is 34 GB. The frequency axis is the
natural chunk: hold the visibilities resident, stream channels through the NUFFT,
and reduce along frequency as you go rather than materialising the cube. Use
`lax.map` over channel chunks with `vmap` inside each, and keep the chunk shape
fixed so compilation is reused; going from 256 to 300 positions retriggers a
73 ms compile against 10 ms cached.

**GPU.** On the cluster this is CUDA, where float64 works. On this Mac it is not
currently available: `jax-metal` last released 0.1.1 in October 2024 against the
jaxlib 0.4.34 plugin interface while the environment has jax 0.11.2, and more
fundamentally Apple GPUs have no double precision, which conflicts with the
float64 decision above. Worth retesting in a throwaway environment rather than
the working one, but do not plan around it.

## 3. Real data: SPT-CL J0459-4947

| | Band 1 | Band 3 |
|---|---|---|
| Pointings | 1 | 7 (mosaic) |
| Coverage | 37.7-45.6 GHz | 84.0-100.0 GHz |
| Channels per window | 128, 128, 128, **960** | 128 x 4 |
| Velocity resolution | 105-121 km/s, **13.7 km/s** | 47-55 km/s |
| Antennas | 50 | 51 |
| Expected lines | ~4 | 11 |

Lines sit at different sky positions in both bands, so this is a genuine blind
search. Each step below is a gate.

### 3.1 Decide whether continuum subtraction is needed
Run the finder on a single pointing with and without `uvcontsub`. A continuum
source biases the filter because the template integrates a smooth component as if
it were line flux. Compare recovered line lists and S/N, then decide whether
subtraction belongs upstream in `alma-data-prep` or here, and which channels are
masked as line-contaminated while fitting.

For the masking decision specifically, a per-channel power statistic works
directly on the visibilities: by Parseval, `sum_j w_j |V_j(nu)|^2` is the power of
the dirty image in channel `nu`, so comparing it with its noise expectation flags
bright channels with no imaging at all. It is spatially integrated, so it dilutes
a faint compact line across the field and will only catch the bright ones, which
is exactly what continuum masking needs.

### 3.2 Detection inference on one pointing
Single Band 3 pointing. Turn the response cube into a catalogue: position,
frequency, width and S/N per candidate. Calibrate the false-positive rate from
the jackknife response over the same grid.

**Jackknife weights (parked).** `jackknife` keeps the pair-averaged weight `w`,
but `(V_a - V_b)/2` has inverse variance `4/(1/w_a + 1/w_b)`, i.e. `2w`. The
jackknife dirty image has the right noise, but the S/N normalisation uses half the
weight, so the jackknife response has variance 1/2 (measured on the fixture:
standard deviation 0.71 against 1.01 for the data). A false-positive rate
calibrated on it is optimistic by `sqrt(2)` in S/N. Fix the weights before this
calibration.

**Grouping and clipping.** One source does not produce one detection. The
response is correlated across neighbouring positions on the beam scale and across
channels on the line-width scale, so a real line lights up a cluster of grid
points, and a bright source can push sidelobe positions over threshold too. Needs
deciding: extract local maxima with an exclusion radius set by the beam and the
line width, or label connected components in (dra, ddec, nu) under that metric.
The false-positive calibration has to count groups rather than grid points,
otherwise the trials factor is badly wrong. Bright-line subtraction before
searching for faint ones belongs here as well.

### 3.3 Joint Band 1 and Band 3 identification
One pointing per band. A line in each band at the same sky position is a redshift
confirmation, since the bands sample different transitions of the same ladder.
Needs co-spatial matching with a tolerance set by the coarser beam, a way to
combine two independent significances, and a catalogue format that spans bands.

### 3.4 All seven Band 3 pointings
Mosaic. Pointings overlapping the same sky position have different primary-beam
attenuation, so visibilities cannot simply be concatenated. Candidate approach: a
per-pointing PB factor in the UV model, and a joint response summed over pointings
with PB-squared weighting. Requires the per-field chunking from section 1.

## 4. Sources that are not Gaussians

The matched filter is only optimal when the template resembles the source, and
everything here assumes an elliptical Gaussian. Lensed systems break that badly:
SDP.81 shows dense-gas tracers and continuum along an Einstein arc, where a single
Gaussian is a poor match and costs real S/N.

Options to weigh: a small basis of components fitted jointly, an explicit arc or
ring parametrisation, or accepting the mismatch and quantifying the S/N lost
against a matched template. Worth measuring the penalty on a simulated arc before
choosing, since the answer decides whether this needs new model classes or only a
caveat in the catalogue.

## Longer term

- **Merge with [alma-data-prep](https://github.com/Joshiwavm/alma-data-prep).**
  Start with the chunked export described in section 1.
- **Bright-line subtraction.** Subtract the best-fit UV model and re-run on the
  residual. Needs a bright-plus-faint fixture. Detection does not need the
  moment-8 machinery in `alma_data_prep.export_cube.ExportCube`: that map is
  `max_nu [I / sigma_nu]`, which is a matched filter with a one-channel boxcar,
  so the filter here already dominates it for a line of known width. Moment-8
  cannot move into the visibility plane either, because `max` is nonlinear and
  pointwise in space while the Fourier relation is linear; Parseval gives total
  power, not a per-pixel maximum. The reason to want it there, avoiding a CASA
  imaging pass, is solved instead by the NUFFT in section 2, which produces the
  dirty cube directly.
- **The `dra` sign flip** between image and model conventions is documented and
  asserted but not fixed. Fixing it means regenerating the committed fixture.
- **Other exploration methods** beyond grid search, once it is understood on real
  data.
