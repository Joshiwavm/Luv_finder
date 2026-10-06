# Roadmap

Current state: grid-search matched filter, in S/N. The data are stored per (field,
spw) as `X = w * V` and a channel flag mask,
with `u`, `v`, time and weight per row and frequency per channel.

The matched filter runs in JAX, float64, on a whole window at once. Per channel,
the position search is one complex matrix product over a `dra x ddec` grid,
with the phase factors advanced from channel to channel by recurrence; the line
template is then centred on every channel and normalised over the channels it
covers. The process pins itself to a quarter of the cores, at most all but two.
The default grid spans the primary beam down to PB = 0.2 in half-resolution
steps.

The per-point code this replaced needed about 8 s per grid point and pass on a
Band 3 pointing, roughly 26 CPU-hours for the same grid.

Primary beam and mosaics. `utils.primary_beam` is an Airy pattern scaled to the
FWHM the ALMA Technical Handbook gives for the real antennas, 1.13 lambda/D (63"
at 92 GHz); simobserve's CASA beam measures 1.165 lambda/D, matched within 1% at
0-35" (`tests/test_primary_beam.py`). A pointing's search is divided by its PB
(S/N unchanged, masked below PB = 0.2) and the pointings are combined with weights
PB^2 / sigma^2 on their shared lattice; their dirty cubes form linear mosaics for
the moment-8 and continuum maps. `search_pointings` searches, corrects and combines every
pointing in one call; the whole Band 3 mosaic, 195 x 183 positions x 3 widths x
512 channels with its jackknife, takes 18 minutes on 12 cores at 13 GB.

From a search to a catalogue (`catalogue.py`, section 2.2): local maxima are
grouped by the response shape measured on the jackknife, and every group is
tested with a likelihood ratio and a fidelity against the jackknife's groups.
The S/N accounts for the correlation between channels that ALMA's Hanning
response leaves, measured on the jackknife. The Band 3 mosaic gives 14 lines
(section 2.2).

The rest of section 1 is about scaling to larger grids.

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

### 2.2 Detection inference
#TODO: double check the effect of continuum subtraction on the line inference.

Open:

- **Completeness.** Inject analytic `Model` lines into the visibilities (ASPECS
  style) and recover them through the same path, to turn the S/N limit into a
  flux limit per coverage (on a mosaic the S/N cut is uniform, the flux limit is
  not).
- **Frequencies in LSRK.** The search works in the measurement set's frame, TOPO
  for these data; at the Band 3 observation dates (March 2024) TOPO to LSRK is
  +7.4 to +7.9 MHz at 85 GHz (~27 km/s), most of the 11 MHz by which the
  catalogue sits below an independent by-eye line list. Catalogue frequencies and
  redshifts should be LSRK, which needs the observation times and the site in
  `Metadata`.
- **Fluxes against image-plane fits** (low priority). For the lines matched to
  that by-eye list, the source fit's line fluxes are 1.4-6.0 times (median 2.8)
  those of Gaussian fits to peak-pixel spectra, which miss the flux of resolved
  lines. Moment-0 maps of the matched lines would show whether they are
  resolved.
- **A fit that absorbs its neighbours.** The candidate at 85.857 GHz, 16-31 MHz
  from two detected lines, is no longer detected (fidelity 0.43), but when it was
  its fit ran to a 1818 km/s line 101 MHz off the peak (chi^2_red 3.5, 10.8
  times the catalogue flux) and still reported `fit_converged`. Fit neighbouring
  lines jointly, or bound the width and centre to the window between them, and
  let a chi^2 or bound check fail the fit.

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
  imaging pass, is solved: `imaging.dirty_cube` produces the dirty cube on the
  search grid by NUFFT, and `dirty_maps` the moment-8 from it.
- **Combining 12 m and 7 m (ACA) data.** The arrays have different dishes, so
  each needs its own primary beam in `pb_corrected`, `combine_pointings` and the
  fit; they share one position lattice, set by the 12 m resolution.
- **Primary beam for extended sources.** The correction is point-like: one PB
  per position. For a source comparable to the beam the PB varies across it;
  folding PB into the template envelope (per pointing) would handle that.
- **Moment maps of found lines.** Moment-0/1 maps of catalogued lines from the
  `imaging.dirty_cube` of their pointings.
- **GPU.** The filter is device-agnostic JAX. miscanti has a Tesla T4 but no
  CUDA jaxlib, and the T4's float64 rate is 1/32 of its float32 rate, so it is
  unlikely to beat the CPU; Apple GPUs have no float64 at all. Worth trying on
  a cluster GPU with real float64 throughput (A100/H100).
- **Other exploration methods** beyond grid search, once it is understood on real
  data.
