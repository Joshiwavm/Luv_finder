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

The matched filter runs in JAX, float64, on a whole window at once. Per channel,
the position search is one complex matrix product over a `dra x ddec` grid,
with the phase factors advanced from channel to channel by recurrence; the line
template is then centred on every channel and normalised over the channels it
covers. A line peaks on its own channel at its S/N: on the size mocks the
template weighting gives 10.000 for a declared 10 at every size, with symmetric
neighbours and no response at the window edges. The process pins itself to a
quarter of the cores, at most all but two. Blind search with the default grid
(+-0.4 primary beam in half-resolution steps, three widths) and its jackknife,
on 12 of 48 cores:

| | Band 1 | Band 3, one pointing |
|---|---|---|
| Grid points x channels | 21,675 x 1,344 | 11,907 x 512 |
| Data, jackknife | 6:39, 3:19 | 0:39, 0:21 |
| Total with loading, peak RSS | 10:55, 22 GB | 1:10, 3.3 GB |

The per-point code this replaced needed about 8 s per grid point and pass on a
Band 3 pointing, roughly 26 CPU-hours for the same grid. Beyond 12 cores the
search does not get faster (16 cores: 43 s against 39 s; 8 cores: 51 s), so the
core budget costs nothing.

Primary beam and mosaics. `utils.primary_beam` is an Airy pattern scaled to the
FWHM the ALMA Technical Handbook gives for the real antennas, 1.13 lambda/D (63"
at 92 GHz); simobserve's CASA beam measures 1.165 lambda/D, matched within 1% at
0-35" (`tests/test_primary_beam.py`). A pointing's search is divided by its PB
(S/N unchanged, masked below PB = 0.2) and the pointings are combined with weights
PB^2 / sigma^2 on their shared lattice; their dirty cubes form linear mosaics for
the moment-8 and continuum maps. The five pointings that cover the 84.57 GHz line
agree in PB-corrected peak flux (0.76-1.06 mJy, chi^2 = 5.9 for 4 degrees of
freedom) and combine to S/N 28.8 (sqrt of the summed S/N^2: 28.9), a line flux of
0.206 +- 0.007 Jy km/s. The whole Band 3 mosaic, 195 x 183 positions x 3 widths x
512 channels with its jackknife, takes 17 minutes on 12 cores at 15.5 GB; it
raises the CASA cube's second line (85.17 GHz, +41", -38") to S/N 24 and several
more candidates to 13-23, against a jackknife maximum of 8.6 (S/N still inflated
by the channel correlation, 2.2).

Section 2 is next: one pointing searches in about a minute, so the real-data
work no longer waits on performance. The rest of section 1 is about scaling to
larger grids.

## 1. Performance

### uv binning

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
  one cell mixes them. Pointings are combined at the response level
  (`combine_pointings`), after `pb_corrected` divides each by its primary beam.
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
positions is therefore computing the dirty image one pixel at a time. The filter
already does that as a matrix product: on a `dra x ddec` grid the phase factorises
into `exp(-2i pi u dra) * exp(-2i pi v ddec)`, so every channel is one complex
GEMM over all positions, `O(n_dra * n_ddec * N_vis)`, and the phase factors
advance from channel to channel by recurrence instead of new exponentials. One
NUFFT per channel would produce every position at roughly
`O(N_vis + N_grid log N_grid)`. It has to be per channel because `u * nu / c`
rescales with frequency, which is ordinary spectral-cube gridding.

**Spectral.** The line template centred on every channel, normalised per lag, is
a matrix over the channel axis (`_spectral`). This part does not change. It runs
on an array of shape `(n_freq, n_pos)`, negligible next to the visibilities.

So a NUFFT would replace only the per-channel GEMM in `_collapse`. One constraint
it adds: the position grid must be regular, where the GEMM takes any `dra` and
`ddec` lists. A blind search wants a regular grid anyway.

Where it starts to matter: the GEMM cost grows with the number of positions times
visibilities, so the Band 1 default grid (21,675 points, 415 k rows in the
960-channel window) takes 6.6 minutes per pass where a Band 3 pointing takes 39 s.

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
4. Dirty map of the continuum, in Jy: `mosaic_dirty_maps` already gives the
   PB-corrected continuum (and moment-8) mosaic on the search grid; save it as
   the continuum diagnostic.

It shares the search's spatial collapse, and the per-(field, spw) chunks bound
its memory.

For the masking decision specifically, a per-channel power statistic works
directly on the visibilities: by Parseval, `sum_j w_j |V_j(nu)|^2` is the power of
the dirty image in channel `nu`, so comparing it with its noise expectation flags
bright channels with no imaging at all. It is spatially integrated, so it dilutes
a faint compact line across the field and will only catch the bright ones, which
is exactly what continuum masking needs.

### 2.2 Detection inference
Turn a search into a catalogue: position, frequency, width, flux and S/N per
candidate, for one pointing or the PB-combined mosaic. The input is a
`SearchResult`: S/N, PB-corrected peak flux density and its error per position x
template x channel, a coverage map, and the jackknife on the same grid.
Calibrate the false-positive rate from that jackknife. On a mosaic the noise
varies across the field (the coverage is `sqrt(sum PB^2)`), so thresholds and
false-positive counts have to follow the coverage rather than be one number.

**Correlated channels.** On real data the response is not yet unit-variance.
The weights are right (whitened jackknife visibilities have variance 1.02), but
adjacent channels of the 128-channel windows are correlated by +0.67 and
next-nearest by +0.17, the Hanning spectral response; the 960-channel Band 1
window, spectrally averaged, shows +0.11. The filter assumes independent
channels, so the jackknife response has a standard deviation of 1.3-1.7 in the
128-channel windows and 1.07 in the 960-channel one. Either put the channel
covariance into the per-lag normalisation of `_spectral` (`sqrt(k^T C k)` with
a tridiagonal-plus `C` per window) or calibrate thresholds on the jackknife per
window; until then, real-data S/N
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

## 3. Sources that are not Gaussians

The matched filter is only optimal when the template resembles the source, and
everything here assumes an elliptical Gaussian, with its major axis at the
position angle `pa` (a grid key, degrees east of north). Nothing keeps
`bmaj >= bmin`, so a grid spanning both orders repeats each shape rotated by 90
degrees. Lensed systems break the Gaussian assumption badly:
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
  imaging pass, is solved: `matchedfilter.dirty_cube` produces the dirty cube on
  the search grid, and `dirty_maps` the moment-8 from it.
- **Primary beam for extended sources.** The correction is point-like: one PB
  per position. For a source comparable to the beam the PB varies across it;
  folding PB into the template envelope (per pointing) would handle that.
- **Moment maps of found lines.** Moment-0/1 maps of catalogued lines from the
  visibilities; install `jax-finufft` (the `jax` extra) for that.
- **GPU.** The filter is device-agnostic JAX. miscanti has a Tesla T4 but no
  CUDA jaxlib, and the T4's float64 rate is 1/32 of its float32 rate, so it is
  unlikely to beat the CPU; Apple GPUs have no float64 at all. Worth trying on
  a cluster GPU with real float64 throughput (A100/H100).
- **The `dra` sign flip** between image and model conventions is documented and
  asserted but not fixed. Fixing it means regenerating the committed fixture.
- **Other exploration methods** beyond grid search, once it is understood on real
  data.
