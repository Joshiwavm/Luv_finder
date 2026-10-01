# Roadmap

Current state: grid-search matched filter, in S/N units. Resolved sources are
handled: `weighting="template"` tapers by the source envelope and recovers the
declared S/N for every `configs/grids/size_*.yaml` preset, with the mock built
from the same grid file (`tests/test_source_size.py`).

The data are stored per (field, spw) as `X = w * V` and a channel flag mask,
with `u`, `v`, time and weight per row and frequency per channel: 17 bytes per
visibility-channel instead of 56. Stokes I comes from `WEIGHT` (set by `statwt`
in alma-data-prep; neither dataset has `WEIGHT_SPECTRUM`) and `FLAG`, and
autocorrelations and flagged rows are dropped on reading. Positions are in one
sky frame per dataset, so a mosaic is searched pointing by pointing on a shared
grid. Measured with `luv-export`:

| | Band 1 | Band 3 |
|---|---|---|
| Visibility-channels kept | 557 M of 588 M | 581 M of 704 M |
| In memory, old layout | 32.9 GB | 39.4 GB |
| In memory, as stored | 9.5 GB | 10.1 GB (1.4 GB per pointing) |
| Export time, peak RSS | 1:36, 14.7 GB | 1:44, 2.6 GB |
| 3 x 3 grid + jackknife, 9 workers | 3:02, 18 GB | 0:17, 7 GB (one pointing) |

Memory for the search is the data, shared copy-on-write by the workers, plus
channel blocks of about 256 MB per temporary in each worker.

Work the two blocks below in order: performance has to land before the real
data are tractable as a blind search.

## 1. Performance

### Analytic evaluation instead of per-grid-point exponentials

Most of the time per grid point is complex exponentials over every
visibility. Because the model is a Gaussian, much of it is redundant. The model
phase is already gone: shifted onto its own position the model is the phase-free
`A(u,v) * S(nu)`, so the kernel is built that way and only the data are
phase-shifted. On Band 1 (557 M visibility-channels, one pointing) a grid point
takes about 70 s per worker.

- **The kernel is position-independent.** `A(u,v; bmin, bmaj)` does not contain
  dra or ddec, so the normalised kernel is identical at every position, verified
  to 3e-14. It needs computing once per (size, width), not once per grid point.
- **The template transform is closed-form.** The Fourier transform of a Gaussian
  is a Gaussian, so `fft(kernel)` can be written down instead of computed.

### Weighted sums as matrix products

The weighted sums are memory-bound. `np.sum(w * t * d, axis=1)` materialises a
full-size temporary and reduces it on one thread; `X @ t` makes one
multithreaded BLAS pass with no temporary. Measured on a 48-core CPU, float64,
128 channels x 500 k visibilities with weights varying per channel, all paths
agreeing to 1e-12:

| Operation | numpy elementwise | numpy `@` | jax `jit`, fused | jax `jit`, `@` |
|---|---|---|---|---|
| Channel collapse, one template | 333 ms | 7.8 ms | 37 ms | 10.7 ms |
| Continuum per row, no `X` | 350 ms | | 20 ms | |
| 64 templates, signal + normalisation | | 220 ms | 7070 ms (`vmap`) | 278 ms |

- `X` is stored, so it is computed once; it does not depend on the template.
  Many templates then stack into one GEMM, about 3.4 ms per template here. The
  stored weight is `w_row` per row times the channel flag mask, so the
  normalisation `w @ t**2` is `(~flag * t**2) @ w_row`.
- `jit` alone gains 17x where no matrix product exists, by fusing the multiply
  into the reduction, but on CPU it does not beat BLAS for `@`.
- Never `vmap` over templates: it materialises the templates x channels x
  visibilities array, as in the JAX section below.

### uv binning, corrected for the primary beam

Averaging visibilities that land in the same uv cell reduces `N_vis`, which is
the dominant axis of both the memory and the cost here.
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
visibility-channels to roughly 18 M, so 39 GB becomes about 1.0 GB.

What it costs, and what has to be right:

- **The cell size is the field of view.** `cell = 1 / FoV` is the usual gridding
  criterion, and sources toward the edge of that field are attenuated by the
  transform of the cell. Measure the S/N loss against source offset on a
  simulated field before committing to a cell size; the usable search radius is
  smaller than the nominal one.
- **Bin within (field, spw), never across pointings.** Visibilities from
  different pointings carry differently PB-weighted skies, so averaging them into
  one cell mixes them. Combining pointings stays at the response level in 2.4.
- **Divide the recovered S/N by PB(theta).** The primary beam attenuates
  off-axis sources within every pointing, so a raw response understates the
  intrinsic line flux away from the phase centre.
- **Do not bin in frequency.** The lines are narrow, and Band 1's 960-channel
  window exists precisely to resolve them.
- `u` and `v` scale with frequency, so a grid fixed in wavelengths is exact at
  one frequency only. Per window the fractional bandwidth is about 2%, so
  gridding at the window centre is accurate to that level. Check it is
  acceptable rather than assuming.
- Bin weight-aware and per channel, `V_cell = sum(X) / sum(w)` with
  `w_cell = sum(w)`, which the stored `X` provides directly.

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

**Fitting the real data.** As stored, a Band 3 pointing is 1.4 GB of
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

## 2. Real data: SPT-CL J0459-4947

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

### 2.1 Decide whether continuum subtraction is needed
Run the finder on a single pointing with and without `uvcontsub`. A continuum
source biases the filter because the template integrates a smooth component as if
it were line flux. Compare recovered line lists and S/N, then decide whether
subtraction belongs upstream in `alma-data-prep` or here, and which channels are
masked as line-contaminated while fitting.

**Continuum subtraction in the uv plane.** The in-package option. The continuum
is estimated in S/N; subtraction and imaging are in Jy:

1. Whiten first: `V' = V * sqrt(w) = X / sqrt(w)`, with `X` and `w` as stored
   per (field, spw).
2. Estimate the continuum of each visibility row along the line of sight,
   over its channels within one (field, spw). A plain mean of `V'` is only
   minimum-variance if `w` is constant across the row, so use the weighted
   mean `sum(w V) / sum(w) = (1 @ X) / (1 @ w)`, which in S/N is `sqrt(w)`
   times that. Leave the masked channels below out of the sum; an unmasked
   line biases it by roughly its width over the window width.
3. Convert back to Jy (divide by `sqrt(w)`) and subtract the continuum from
   every channel of the row.
4. Dirty map of the continuum, in Jy, with `jax-finufft`: the type-1 NUFFT of
   the continuum visibilities with natural weights, `sum(w C e^{...}) /
   sum(w)`. Multiply it by the primary beam and save it as the continuum
   diagnostic.

This shares the NUFFT with the position search in section 1, and the
per-(field, spw) chunks bound its memory.

For the masking decision specifically, a per-channel power statistic works
directly on the visibilities: by Parseval, `sum_j w_j |V_j(nu)|^2` is the power of
the dirty image in channel `nu`, so comparing it with its noise expectation flags
bright channels with no imaging at all. It is spatially integrated, so it dilutes
a faint compact line across the field and will only catch the bright ones, which
is exactly what continuum masking needs.

### 2.2 Detection inference on one pointing
Single Band 3 pointing. Turn the response cube into a catalogue: position,
frequency, width and S/N per candidate. Calibrate the false-positive rate from
the jackknife response over the same grid.

**Correlated channels.** On real data the response is not yet unit-variance.
The weights are right (whitened jackknife visibilities have variance 1.02), but
adjacent channels of the 128-channel windows are correlated by +0.67 and
next-nearest by +0.17, the Hanning spectral response; the 960-channel Band 1
window, spectrally averaged, shows +0.11. The filter assumes independent
channels, so the jackknife response has a standard deviation of 1.3-1.7 in the
128-channel windows and 1.07 in the 960-channel one. Either put the channel
covariance into the kernel normalisation (a tridiagonal-plus term per window)
or calibrate thresholds on the jackknife per window; until then, real-data S/N
is inflated by up to ~1.6. The jackknife weight itself is fixed: the pair
difference carries `4/(1/w_a + 1/w_b)`.

**Grouping and clipping.** One source does not produce one detection. The
response is correlated across neighbouring positions on the beam scale and across
channels on the line-width scale, so a real line lights up a cluster of grid
points, and a bright source can push sidelobe positions over threshold too. Needs
deciding: extract local maxima with an exclusion radius set by the beam and the
line width, or label connected components in (dra, ddec, nu) under that metric.
The false-positive calibration has to count groups rather than grid points,
otherwise the trials factor is badly wrong. Bright-line subtraction before
searching for faint ones belongs here as well.

### 2.3 Joint Band 1 and Band 3 identification
One pointing per band. A line in each band at the same sky position is a redshift
confirmation, since the bands sample different transitions of the same ladder.
Needs co-spatial matching with a tolerance set by the coarser beam, a way to
combine two independent significances, and a catalogue format that spans bands.

### 2.4 All seven Band 3 pointings
Mosaic. Pointings overlapping the same sky position have different primary-beam
attenuation, so visibilities cannot simply be concatenated. Candidate approach: a
per-pointing PB factor in the UV model, and a joint response summed over pointings
with PB-squared weighting. The shared grid exists: positions are offsets from one
reference direction, each field's phase centre is stored in that frame, and
`luv-find --field` searches one pointing on lattice points common to all of them.
Checked on the brightest line of the CASA dirty mosaic cube (84.57 GHz, 2" east
and 20.5" north of field 0): every pointing recovers it at that WCS position to
within 1-2", at S/N 14-17 in the three pointings ~20" away and falling with
distance as the primary beam does, so the model's `dra` is east on real data.
What remains is the PB factor and the combination.

## 3. Sources that are not Gaussians

The matched filter is only optimal when the template resembles the source, and
everything here assumes an elliptical Gaussian. It does not even have a position
angle yet: `bmaj` lies along RA and `bmin` along Dec, in the model and the mock
alike, and nothing keeps `bmaj >= bmin`. Add `pa` as a grid key (rotate u, v in
`Gaussian.envelope`, `theta` in the mock's `Gaussian2D`) before searching for
elongated sources. Lensed systems break the Gaussian assumption badly:
SDP.81 shows dense-gas tracers and continuum along an Einstein arc, where a single
Gaussian is a poor match and costs real S/N.

Options to weigh: a small basis of components fitted jointly, an explicit arc or
ring parametrisation, or accepting the mismatch and quantifying the S/N lost
against a matched template. Worth measuring the penalty on a simulated arc before
choosing, since the answer decides whether this needs new model classes or only a
caveat in the catalogue.

## Longer term

- **Merge with [alma-data-prep](https://github.com/Joshiwavm/alma-data-prep).**
  The per-(field, spw) reader in `data.py` is the piece to share: Stokes I from
  `WEIGHT` and `FLAG`, the selection alma-data-prep's own export applies.
- **Bright-line subtraction.** Subtract the best-fit UV model and re-run on the
  residual. Needs a bright-plus-faint fixture. Detection does not need the
  moment-8 machinery in `alma_data_prep.export_cube.ExportCube`: that map is
  `max_nu [I / sigma_nu]`, which is a matched filter with a one-channel boxcar,
  so the filter here already dominates it for a line of known width. Moment-8
  cannot move into the visibility plane either, because `max` is nonlinear and
  pointwise in space while the Fourier relation is linear; Parseval gives total
  power, not a per-pixel maximum. The reason to want it there, avoiding a CASA
  imaging pass, is solved instead by the NUFFT in section 1, which produces the
  dirty cube directly.
- **The `dra` sign flip** between image and model conventions is documented and
  asserted but not fixed. Fixing it means regenerating the committed fixture.
- **Other exploration methods** beyond grid search, once it is understood on real
  data.
