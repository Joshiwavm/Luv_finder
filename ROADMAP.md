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

The line fit is the easiest place to start. It moves within about a resolution of
the matched filter's peak and its sources are a few arcsec, so after a phase
rotation to the peak the cells can be large: 5 klambda loses
`sinc(pi 5 klambda 2")` ~ 0.4% at 2" in Band 3, and leaves a few hundred to a
thousand cells of 196 k rows. Each pass of the fit (`kernel.point_moments`) would be
100-200x cheaper; worth it for Band 1's 960-channel window or larger mosaics.

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
- **Fluxes against the by-eye list** (low priority). That list is the published
  one: lines found by eye in the image cubes, their fluxes from Gaussian fits to
  the extracted spectra. The uv fit's line fluxes are 1.4-6.0 times (median 2.8)
  those, so this compares that simple image-plane method with the full fit; the
  moment-0 maps (`imaging.line_maps`) show which lines are extended, where a
  spectrum misses flux.
- **Atmospheric lines.** Atmospheric lines make some channels noisier than a
  weight per spectral window says: ozone at 1.2 mm (231.3, 235.7, 237.1, 239.1 GHz
  and more above 240 GHz) and the 118.75 GHz O2 wing at the top of Band 3. With
  `statwt chanbin="spw"` the S/N there is overestimated in the data and the jackknife
  alike; in ASPECS 1.2 mm the excess is 1.5-3.4x in power and changes by a factor of
  two between execution blocks. Done: the weights are measured on reading per
  (scan, channel) from integrations of a baseline <= 450 s apart, any field
  (`weight_scale`, 99% of the rows paired at 1.2 mm), and divided by it. Per scan,
  because the 237.15 GHz ozone core also changes between the scans of a block, by
  +-8% in power in 3 of 8 blocks (chi^2/dof 2.4-7.4 at the line peak; clean channels
  1.0). With the weights as read, false peaks gathered on the ozone lines (3-12x
  the chance rate at S/N >= 5); corrected per channel, they no longer do.
- **Narrow features that change in time.** The robust check of `weight_scale` exceeds
  5 sigma in ASPECS 3 mm spw 6, 7, 9 and 11. In spw 9 (20.7 sigma) it is two channels
  at 108.88 GHz in two execution blocks, with heavy-tailed pair differences on many
  baselines (the top 1% of pairs carry 11% of `sum w z^2`, twice the Gaussian share):
  interference or a spur. The mean-based scale down-weights those cells by their
  real variance; flagging the (scan, channel) cells the check marks would be cleaner.
- **Jackknife of sparsely sampled fields.** A mosaic field seen once per visit
  (ASPECS 1.2 mm: 9-14 integrations per window, days to months apart) pairs fewer
  rows than one tracked for hours: the per-baseline pairing keeps 67-74% of them
  (98% at 3 mm), down from 33-42% when integrations were paired by their index in
  time. Partners far apart in hour angle cancel the sky less well; the continuum
  fit removes leaked continuum, but a bright line could leak. Check the leakage on
  a mock observed the same way. At 1.2 mm the data hold 4-9% more groups than the
  jackknife at S/N 4-5.5, while the negated data match the data: a pair sits at the
  mean of its partners' uv, which differ when they are months apart. Compare with a
  jackknife that keeps one partner's uv.
- **Fit across overlapping windows.** Windows that overlap in frequency are combined
  in the search, but `fit_line` uses one window per pointing, so a line in the
  ASPECS 3 mm double-tuned range is fitted with one tuning's data.
- **Lines across a window edge.** A line wider than the gap between two adjacent
  windows is searched in each window apart, so it is found twice: ASPECS 3mm.07 (609
  km/s at 93.558 GHz, between spw00, ending at 93.563 GHz, and spw01, starting at
  93.570 GHz) is detected as its low half in spw00 (fitted 476 km/s) and its high
  half in spw01 (243 km/s), 0.4" and 300 km/s apart, each with part of the flux.
  Done: `run.duplicates` calls two detections one line when they are within a beam
  and their fitted profiles overlap, `|dv| <= (FWHM_a + FWHM_b) / 2`, so the fainter
  half is flagged. Open: search and fit adjacent windows as one spectrum (a template
  and a line profile across the gap), so the line is found once with all its flux.
- **Channel response.** The template is a Gaussian sampled at the channel centres;
  Hanning smoothing and channel averaging broaden a line a few channels wide, and
  windows with different correlator setups then see slightly different peaks. The
  fit has the same gap: a line two channels wide (ASPECS 3mm.08, 50 km/s) can
  collapse below one channel, to a width of 4 km/s with an unbounded error. A lower
  bound on the width near the channel width, or the channel response in the model,
  would stop it.
- **Fidelity per template width.** ASPECS computes the fidelity per kernel width:
  narrow templates have more independent trials than broad ones at the same S/N.
  The catalogue pools all templates. Its noise is a Gaussian tail fitted to the
  jackknife groups, as ASPECS fits its negatives; an erf fitted to the binned
  fidelity failed at 1.2 mm, where the jackknife deficit near the floor (above)
  held the low bins at 0.05 and set the threshold at S/N 7.9 instead of 6.7.
- **A fit that absorbs its neighbours.** The candidate at 85.857 GHz, 16-31 MHz
  from two detected lines, is no longer detected (fidelity 0.43), but when it was
  its fit ran to a 1818 km/s line 101 MHz off the peak (chi^2_red 3.5, 10.8
  times the catalogue flux) and still reported `fit_converged`. Fit neighbouring
  lines jointly. Cutting the fit to a few widths around the peak does not help:
  in so few channels the quadratic continuum mimics a broad line (tried: one line
  ran to 5800 km/s, another lost 30% of its flux).

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

## 4. Publication

Two papers, submitted together (JOSS supports co-publication with a methods paper):

| Paper | Venue | Content |
|---|---|---|
| Methods | A&A, regular paper (2024 impact factor 6.1) | The method; validation on simulations and on ASPECS; SPT-CL J0459-4947 as the application |
| Software | JOSS, 1-2 pages | Summary, statement of need, design; cites the methods paper for the math |

Considered and set aside: RASTI (impact factor 2.5); an A&A Letter (four main-body
pages since April 2025, and it needs a timely result); an AAS journal with JOSS's
joint review.

The methods paper validates on ASPECS, the cleanest test: public data with a published
blind catalogue (Gonzalez-Lopez et al. 2019). It then compares the SPT-CL J0459 Band 3
mosaic (14 lines) and Band 1 with our published by-eye list. Before submitting:

- the ASPECS comparison: recovered, new and missed lines, and the S/N and flux ratios;
- completeness, LSRK frequencies and joint fits of neighbouring lines (section 2.2);
- the Band 1 search and the joint Band 1 and Band 3 identification (section 2.3).

JOSS gates, all needed at submission:

- More than six months of public, iterative development. The history starts on
  2026-09-22, so late March 2027 at the earliest.
- Research use, shown by the methods paper on arXiv.
- Tagged releases with an archived DOI (Zenodo).
- `CHANGELOG.md`, `CONTRIBUTING.md` and `CITATION.cff` (in place).
- An AI usage disclosure that the author confirms.

Both drafts live in Notion: the method chapter on the Luv-finder page, and the JOSS
article as its subpage. The `notion-sync` skill (`.claude/skills/notion-sync/`) keeps
them in step with `main`.

## 5. Archive mining

Goal: a blind line search over the whole public ALMA archive. Find the data that
add the most survey volume, download only what calibration needs, restore it
with the right CASA, prepare it with alma-data-prep, and search it here.

### What exists

`scripts/alma_sky_coverage.py` (not part of the package) queries the archive's
`ivoa.obscore` through ALminer in (band, Dec strip) chunks, cached in
`data/archive/chunks/`, and collapses the per-spw rows to one row per (MOUS,
footprint): band, footprint, array (12m/7m), data rights, the frequency coverage
as merged spw intervals and the summed bandwidth. `plot` draws the coverage
figure. The hands-on notebook `04_alma_archive_coverage` in
`ALMA_analysis/Notebooks/luv_finder/` runs both and selects by field, band, line
and redshift, array and data rights, giving one row per MOUS (project code, MOUS,
targets, area, bandwidth) as a CSV.

State of the archive on 2026-10-14 (ESO mirror, Dec < +30, science targets,
interferometric only): 430,977 rows, 87,734 footprints, fetched in 8 minutes.

| | Footprints | Unique area [arcmin^2] | Band 3 | Band 6 |
|---|---|---|---|---|
| All science targets | 87,734 | 58,619 | 39,343 | 18,656 |
| Extragalactic | 43,628 | 15,431 | 8,795 | 4,936 |

Extragalactic means `Galaxy evolution`, `Active galaxies` and `Cosmology`. About
half the equatorial strip is COSMOS; the rest is H-ATLAS/GAMA (9h-15h) and SDSS
Stripe 82 (20h-4h). Band 2 has no science data yet.

### What the archive tables are like

Checked on real rows; each point changed the script:

- `s_region` is STC-S: `Circle` for a single pointing (radius `s_fov / 2`),
  `Polygon` for a mosaic, often non-convex (healpy's `query_polygon` fails on
  about 20% of them), `Union ( ... )` of disjoint outlines, and holes as
  `Not (Polygon ...)`. Footprints are rasterised by point-in-polygon on the
  candidate pixels of a bounding disc, with the holes removed.
- `s_fov` is the PB FWHM for a single pointing, but for a `Union` it spans all
  members, up to 38 degrees. Areas come from the rasterised outlines, never from
  `s_fov`.
- `scientific_category` now reads `Galaxy evolution`, `Active galaxies`,
  `Cosmology`, `Local Universe`, `ISM and star formation`, ...; the older names
  match nothing. `Active galaxies` includes Sgr A* and the CMZ (about 400
  footprints), and 2015.1.01289.S files 1,179 footprints at l = 350, b = 0 as
  `Galaxy evolution`: an extragalactic selection still needs a Galactic-plane cut.
- `antenna_arrays` lists `pad:antenna`: DA/DV are 12 m, CM 7 m, PM total power.
- `em_min`/`em_max` are wavelengths in metres. `band_list` has multi-band values
  ("3 7") on 103 of about 449,000 science rows; an exact match skips them.
- `s_region` is a geometry column on the ESO TAP service, so `LIKE` on it fails.

ALminer pins pyvo 1.2.1, which has no `DALOverflowWarning` (a truncated result
passes silently) and imports `cgi` (gone in Python 3.13), and an astroquery that
needs `setuptools<81`. Its `download_data` breaks with current pandas (it hands
astroquery a string array, not a list). It runs in its own venv
(`~/venvs/almacov`, Python 3.12, pyvo upgraded), never in `luv`.

### From a MOUS to calibrated visibilities

`astroquery.alma.Alma.get_data_info([mous])` lists every file of a MOUS, by
`semantics`:

| Semantics | What | Needed for visibilities |
|---|---|---|
| `#this` | `<project>_<mous>_001_of_001.tar`: pipeline images and masks | no |
| `#auxiliary` | `<project>_<mous>_auxiliary.tar`: `scriptForPI.py`, `caltables.tgz`, `calapply.txt` per EB, flag versions, the pipeline manifest, weblog, QA reports | yes |
| `#progenitor` | `<project>_<EB>.asdm.sdm.tar`: raw data of the calibrated EBs | yes |
| `#package` | further ASDMs without a `calapply.txt`, not calibrated (QA0 failures, presumably) | no |
| `#documentation` | README | yes |

With `expand_tarfiles=True` the members are listed one by one, and small ones
(README, `scriptForPI.py`, the manifest) can be read without downloading a tar.
The CASA and pipeline versions are in
`calibration/*pipeline_manifest.xml` (`<casaversion name=...>`,
`<pipeline_version name=...>`), not in the README. Restoring: untar everything in
one directory, which recreates
`<project>/science_goal.uid___*/group.uid___*/member.uid___*/{raw,calibration,script,...}`,
then in `script/` run `casa --pipeline -c member.uid___*.scriptForPI.py` with that
version. The calibrated MSs land in `calibrated/` (`*.ms.split.cal`, or
`*_targets.ms` for pipeline-imaged data). That tree is what alma-data-prep's
`ProjectDataOrganizer` walks for `*.ms.split.cal`, so it needs no change.
Alternatively the EU ARC delivers calibrated MSs on request (Helpdesk ticket).

Allegro has, after `allegro-setup`: `casapy431`, `casapy452`, `casapy460`,
`casapy470`, `casapy-472`, `casapy-500` ... `casapy-580`, `casapy-611p`,
`casapy-621p`, `casapy-630`, `casapy-641p`, `casapy-654p`, `casapy-655`,
`casapy-666p`, `casapy-672`. CASA `X.Y.Z.*` maps to `casapy-XYZ`, and the `p`
builds carry the pipeline; restoring pipeline-calibrated data needs a `p` build.
At archive scale the version bookkeeping, not the download, is the work.

Worked example, 2023.1.00213.S (PI Kitayama), J0459.6-4946:

| | |
|---|---|
| MOUS | `uid://A001/X3621/X376d` (the project's only one); group `X3621/X376c`, science goal `X3621/X376b` |
| Setup | Band 3, 12 m mosaic, 4 spws from 84.04 GHz, 31.25 MHz channels, about 2.8" |
| On source | 11,154 s; calibrators J0519-4546, J1107-4449 (bandpass, flux), J0451-4653, J0455-4615, J0526-4830, J0531-4827 (phase) |
| Rights | public since 2025-04-16, QA2 passed |
| CASA | 6.5.4.9, pipeline 2023.1.0.124: `casapy-654p` |
| EBs | 15 calibrated, 4 more `#package` ASDMs not |
| Download for visibilities | auxiliary tar 3.0 GB + 15 ASDMs 127 GB (images tar 0.28 GB, not needed) |

All 15 calibrated EBs are already in
`~/Documents/Allegro-home/open_CLJ0459/archive/2023.1.00213.S/.../member.uid___A001_X3621_X376d/calibrated/`.

### Plan: Luv_finder, ALminer and alma-data-prep run jointly

A `luv_finder.archive` module and a `luv-archive` CLI, on pyvo and astroquery
directly rather than ALminer (its pins break `luv`, its download is broken, and
it is a thin layer over the two), keeping ObsCore column names so ALminer
functions still read the tables:

1. `luv-archive query`: the footprint and MOUS table, moved from
   `scripts/alma_sky_coverage.py` into the package, with the selections of the
   notebook (field, band, line and redshift, array, data rights, a Galactic-plane
   cut).
2. `luv-archive manifest <MOUS ...>`: per MOUS the archive tree path, the
   calibrated EBs, the sizes and the CASA and pipeline versions from the
   manifest, mapped to the `casapy` build. Writes `download.sh` (`wget -c` of the
   auxiliary tar and the `#progenitor` ASDMs; public data only, proprietary data
   need an archive login) and `restore.sh` (untar, then `scriptForPI.py` under
   the right `casapy` per MOUS).
3. alma-data-prep: `ProjectDataOrganizer(<archive root>)` as now, to a
   concatenated MS.
4. Luv_finder: the `luv-run` YAML points `data:` at that MS; later a run could
   name a project or MOUS instead.

Open:

- Where `archive` lives: here (the search defines what is worth downloading) or
  in alma-data-prep (getting data is its job, but it runs in CASA's python, which
  may lack pyvo).
- `restore.sh` needs `allegro-setup` and a `casapy`, so it belongs on the CASA
  side, with alma-data-prep.
- The figures' numbers move with the archive (new data monthly, proprietary data
  turning public); a release of the table should record its query date.

## Longer term

- **Merge with [alma-data-prep](https://github.com/Joshiwavm/alma-data-prep).**
  The per-(field, spw) reader in `data.py` is the piece to share: Stokes I from
  `WEIGHT` and `FLAG`, the selection alma-data-prep's own export applies.
  Archive mining (section 5) joins the two from the other end.
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
