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
0.206 +- 0.007 Jy km/s. `search_pointings` searches, corrects and combines every
pointing in one call; the whole Band 3 mosaic, 195 x 183 positions x 3 widths x
512 channels with its jackknife, takes 18 minutes on 12 cores at 13 GB.

From a search to a catalogue (`catalogue.py`, section 2.2): local maxima are
grouped by the response shape measured on the jackknife, and every group is
tested with a likelihood ratio and a fidelity against the jackknife's groups.
Correlated channel noise is not accounted for anywhere yet (section 2.2,
open). The Band 3 mosaic gives 16 lines (section 2.2).

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

### 2.1 Continuum (done)
Each line template is fitted jointly with a polynomial continuum (default degree
2) per window at every trial position: generalised least squares, profiling the
continuum out of the line amplitude (`_spectral`, `continuum_order`). Per position,
because after the phase shift a continuum source at the trial position is smooth
in frequency however far from the phase centre; a fit per visibility row, as
CASA's uvcontsub, fails there (the phase winds by 2.5 rad across a 2 GHz Band 3
window at 30" on the longest baselines). Jointly, because it is linear and exactly
normalised and needs no line-free channel selection: the trial line's own
channels are in its model. A first version clipped bright bins and refitted; it
was non-linear in the noise (the fit absorbed the core while clipped excursions
kept their height), which fattened the jackknife's tail relative to its core
(local maxima beyond 6 core sigma: 0 -> 7 on the mosaic). The joint fit leaves
the noise Gaussian and costs no time. On Band 3 it removes the SZ decrement's
leak into the line search (-4.5 core sigma before) and the dust continuum under
the 84.57 GHz line (its S/N drops by 8%).

Cost: a line spends some S/N on the continuum's degrees of freedom, more near a
window's edge and in short windows (30 channels: 18%; 128 channels: a few %,
more at the edges). A bright line also leaves a broad response at the other
lags of its own window at the position (2-20% of its S/N), which the noise
correlation includes.

### 2.2 Detection inference (done; open items below)
`catalogue(result)` turns a `SearchResult` (one pointing or the PB-combined
mosaic, with its jackknife) into an astropy table, written as ECSV
(`luv-find --jackknife --catalogue`). The jackknife has the data's noise and no
sky, and is used twice. The S/N is the filter's, uncorrected for correlated
channel noise (see Open).

1. **Response shape.** The null correlation rho of the filter output is also the
   expected response to a matched line (Vio & Andreani 2021). Measured on Band 3:
   half-power ellipse 4.9" x 2.9" FWHM (4.7" x 2.6" for the 100 km/s
   template), sidelobes <= 0.03 (so a 17-sigma line
   puts < 1 sigma into its sidelobes), templates of 200/300/400 km/s correlated
   by 0.97/0.90/0.98 as sqrt(2 W1 W2 / (W1^2 + W2^2)) predicts, and every bright
   line follows `S/N rho`.
2. **Grouping and testing.** Local maxima (3x3x3 in position and channel) of the
   best-template S/N above 4, merged brightest first inside the brighter one's
   half-power ellipsoid of rho; on Band 3 every bright line is one maximum even
   within 3x that ellipsoid. Data, jackknife and (diagnostic only) negated data
   are grouped alike. Each data group gets the likelihood ratio
   `N_data(>= S/N) / N_jk(>= S/N)` (van Marrewijk et al. 2025), required >= 3,
   a lower limit above the jackknife's brightest group, and the fidelity
   `1 - N_jk / N_data` per S/N bin fitted with an error function (Walter et al.
   2016, the jackknife instead of negatives), required >= 0.6.

Band 3 mosaic, templates of 100/200/300/400 km/s: 9450 data groups against 9619
jackknife groups above 4 (jackknife maximum 8.31); the likelihood ratio reaches
3 at S/N 7.85 and the fidelity 0.6 at 7.96. Divided by the jackknife spread
(1.38-1.62 per template) that is ~5 sigma, as the paper found for broad scans.
16 lines pass; the four brightest are at S/N 19-27 (12-17 after that division),
three of them in the CASA cube. Three of the weaker ones sit near window edges
(84.15, 85.83, 85.84 GHz) and lost most to the continuum fit in an earlier run
with three templates; worth inspecting.

**Source fit** (`fit.py`, `luv-find --catalogue --fit`). Detection stays a grid
search; each catalogued line is then refined by a deterministic least-squares fit
in the visibilities, no samplers: a Gaussian source whose spectrum is a Gaussian
line on a degree-2 continuum of the same shape, times each pointing's primary
beam at the source, jointly in every pointing with PB >= 0.2, in the line's
window. chi^2 over the visibilities reduces exactly (to machine precision in the
test) to each pointing's template-weighted spectrum at the trial position and
shape, the filter's own collapse at one point, so the spectrum is a linear fit
plus (nu0, width) and the position and shape a 5-parameter Nelder-Mead. On the
Band 3 mosaic the 16 lines take 25 minutes on 12 cores (up to 7 pointings; the
whole notebook peaks at 19.5 GB with the NPZ loaded whole). Fifteen fit well,
chi^2_red 0.72-1.18: positions move by 0.1-0.7", frequencies by up to 12 MHz.
Seven are resolved along one axis at >= 2.4 sigma (FWHM 1.6-3.0") and
unresolved along the other, five keep a Gaussian consistent with a point and
three collapse to a point. Their line fluxes are 0.95-1.67 times the
catalogue's (median 1.21), most where the fit resolves the source or widens the
line beyond its template (lines at 572 and 621 km/s against 400), so they move
away from the image-plane fits below, not towards them. The sixteenth fails
while reporting convergence (see Open). Errors are formal (see correlated
channel noise below).

Open:

- **Completeness.** Inject analytic `Model` lines into the visibilities (ASPECS
  style) and recover them through the same path, to turn the S/N limit into a
  flux limit per coverage (on a mosaic the S/N cut is uniform, the flux limit is
  not).
- **One jackknife realisation.** The tail beyond its brightest group is only
  bounded (lower limits on the likelihood ratio) or extrapolated (the fidelity
  fit). Our jackknife differences consecutive integrations, deterministically.
- **Frequencies in LSRK.** The search works in the measurement set's frame, TOPO
  for these data; at the Band 3 observation dates (March 2024) TOPO to LSRK is
  +7.4 to +7.9 MHz at 85 GHz (~27 km/s), most of the 11 MHz by which the
  catalogue sits below an independent by-eye line list. Catalogue frequencies and
  redshifts should be LSRK, which needs the observation times and the site in
  `Metadata`.
- **Fluxes against image-plane fits.** For the lines matched to that by-eye list,
  the line fluxes here (point-source template of fixed width, fitted in the uv
  plane, PB-corrected per pointing) are 1.3-4.8 times those of Gaussian fits to
  extracted spectra, and the source fit's 1.4-6.0 times (median 2.8); the ratio
  does not follow the PB coverage. Not understood yet.
- **Correlated channel noise: not accounted for.** ALMA's Hanning spectral
  response correlates neighbouring channels (+0.67 adjacent, +0.17 next on Band
  3). Nothing in the analysis models it yet: the filter's S/N and the errors of
  the catalogue and of the source fit all assume independent channels, so the
  noise is underestimated and S/N values are too high (the jackknife's S/N spread
  is 1.38-1.62 per template on Band 3 instead of 1;
  `catalogue.jackknife_spread` reports it).
  Detection is not biased by it: the likelihood ratio and the fidelity only
  compare the data with the jackknife, which carries the same correlated noise,
  so its effect drops out there; but every S/N and threshold quoted is in these
  inflated units. To understand before modelling it (the covariance `C^-1` in
  the template, Vio & Andreani 2021, and in the fit errors).
- **A fit that absorbs its neighbours.** Line 15 (85.857 GHz) lies 16-31 MHz
  from lines 8 and 9; its fit runs to a 1818 km/s line 101 MHz off the peak
  (chi^2_red 3.5, 10.8 times the catalogue flux) and still reports
  `fit_converged`. Fit neighbouring lines jointly, or bound the width and centre
  to the window between them, and let a chi^2 or bound check fail the fit.
- **Bright lines.** Only needed once a field has lines bright enough for their
  sidelobes (> ~50 sigma) or their window-wide continuum-fit response to cross
  the floor; see bright-line subtraction under Longer term.

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
