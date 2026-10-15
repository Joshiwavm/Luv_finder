"""ALMA's interferometric footprint on the southern sky, from the archive's ObsCore table.

A discussion figure for an archive-wide blind line search, not part of the package:

    python scripts/alma_sky_coverage.py fetch                  # -> data/archive/alma_fields.ecsv
    python scripts/alma_sky_coverage.py plot [--extragalactic] # -> plots/alma_sky_coverage.png

``fetch`` queries ``ivoa.obscore`` through ALminer (``alminer.run_query``) in (band, Dec strip)
chunks, cached in ``data/archive/chunks/`` so an interrupted fetch resumes, and collapses the
per-spw rows to one row per (member OUS, footprint): band, footprint, the frequency coverage as
merged spw intervals, and the summed bandwidth. ``plot`` draws

- the whole sky (Mollweide, equatorial; north of ``--dec-max`` greyed) with the area ALMA covers per HEALPix
  cell: the footprints are rasterised on a fine HEALPix grid (``--nside``, 3.2" pixels by
  default) and the unique covered pixels summed per coarse cell, so repeat visits count once;
- zooms on a few deep fields with every footprint drawn: ``s_region`` is the half-power circle
  (``s_fov`` = PB FWHM) of a single pointing and the outline of a mosaic, often non-convex,
  with ``Not (...)`` holes, or a ``Union`` of disjoint outlines (whose ``s_fov`` spans them all,
  up to tens of degrees, so areas come from the rasterised outlines, never from ``s_fov``);
- per band, the unique area and area x bandwidth (the blind-line survey volume before redshift),
  the latter summed over footprints, so a field seen twice in one tuning counts twice.

ALminer pins pyvo 1.2.1, which has no ``DALOverflowWarning`` (a truncated result passes
silently) and imports ``cgi`` (gone in Python 3.13): ``pip install -U pyvo`` after ALminer;
its old astroquery also needs ``setuptools<81`` for ``pkg_resources``. Needs ``healpy``.
"""

import argparse
import re
import warnings
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MplPath

C = 299792458.0  # m/s
FIELDS = Path("data/archive/alma_fields.ecsv")
COLUMNS = (
    "member_ous_uid, proposal_id, target_name, s_ra, s_dec, s_fov, s_region, band_list, em_min, em_max, "
    "is_mosaic, scientific_category, antenna_arrays, data_rights, publication_year"
)
EXTRAGALACTIC = ("Cosmology and the high redshift universe", "Galaxies and galactic nuclei", "Active galaxies")
ZOOMS = {  # name: (ra, dec, half-width) in deg
    "ECDFS": (53.12, -27.81, 0.45),
    "COSMOS": (150.12, 2.21, 0.9),
    "UDS": (34.40, -5.20, 0.55),
}
BAND_COLORS = {
    1: "#6e2a8c",
    2: "#3b4cc0",
    3: "#1f77b4",
    4: "#17becf",
    5: "#2ca02c",
    6: "#bcbd22",
    7: "#ff7f0e",
    8: "#d62728",
    9: "#e377c2",
    10: "#7f7f7f",
}


# ---------------------------------------------------------------- fetch


def fetch(out=FIELDS, tap_service="ESO", dec_max=30.0, strip=10.0, public=False):
    """Query ObsCore per (band, Dec strip) through ALminer and collapse to one row per footprint."""
    import alminer
    from pyvo.dal.exceptions import DALOverflowWarning

    cache = out.parent / "chunks"
    cache.mkdir(parents=True, exist_ok=True)
    rights = " AND data_rights = 'Public'" if public else ""
    parts = []
    for band in range(1, 11):
        for lo in np.arange(-90.0, dec_max, strip):
            hi = min(lo + strip, dec_max)
            path = cache / f"band{band}_dec{lo:+.0f}.ecsv"
            if not path.exists():
                query = (
                    f"SELECT {COLUMNS} FROM ivoa.obscore WHERE band_list = '{band}' "
                    f"AND s_dec >= {lo} AND s_dec < {hi} AND science_observation = 'T' "
                    f"AND scan_intent LIKE '%TARGET%'{rights}"
                )
                with warnings.catch_warnings():  # a truncated result must not pass silently
                    warnings.simplefilter("error", DALOverflowWarning)
                    df = alminer.run_query(query, tap_service=tap_service)
                Table.from_pandas(df.astype({"publication_year": str})).write(path, overwrite=True)
                print(f"band {band:2d}, Dec [{lo:+.0f}, {hi:+.0f}): {len(df)} rows")
            parts.append(Table.read(path))
    fields = collapse([t for t in parts if len(t)])
    fields.write(out, overwrite=True)
    print(f"{len(fields)} footprints -> {out}")
    return fields


def collapse(parts):
    """One row per (member OUS, footprint) from ObsCore's one row per spw; total-power-only rows dropped."""
    from astropy.table import vstack

    rows = vstack(parts).to_pandas()
    interferometric = rows["antenna_arrays"].fillna("").str.contains(r"\b(?:DA|DV|CM)\d", regex=True)
    rows = rows[interferometric | (rows["antenna_arrays"].fillna("") == "")]
    rows = rows.assign(nu_lo=C / rows["em_max"] / 1e9, nu_hi=C / rows["em_min"] / 1e9)  # em_* in m, nu in GHz
    out = []
    for (mous, region), g in rows.groupby(["member_ous_uid", "s_region"], sort=False):
        spans = merge_intervals(g[["nu_lo", "nu_hi"]].to_numpy())
        first = g.iloc[0]
        out.append(
            dict(
                member_ous_uid=mous,
                proposal_id=first["proposal_id"],
                target_name=first["target_name"],
                band=int(str(first["band_list"]).split()[0]),
                ra=first["s_ra"],
                dec=first["s_dec"],
                fov=first["s_fov"],
                s_region=region,
                is_mosaic=str(first["is_mosaic"]) == "T",
                category=first["scientific_category"],
                nu_spans=" ".join(f"{a:.4f}-{b:.4f}" for a, b in spans),
                bandwidth=float(np.sum(spans[:, 1] - spans[:, 0])),
            )
        )
    return Table(rows=out)


def merge_intervals(spans):
    spans = spans[np.argsort(spans[:, 0])]
    merged = [spans[0].copy()]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append(np.array([a, b]))
    return np.array(merged)


# ---------------------------------------------------------------- footprints


def shapes(s_region):
    """``s_region`` (STC-S: Circle, Polygon, Union of them, ``Not (...)`` holes) as a list of
    ("circle", ra, dec, r) | ("polygon", ras, decs), with "hole" for a polygon inside ``Not``."""
    out = []
    pattern = r"(not\s*\(\s*)?(circle|polygon)\s+(?:icrs\s+)?([-+\d.eE\s]+)"
    for neg, kind, nums in re.findall(pattern, s_region, flags=re.I):
        x = np.array(nums.split(), dtype=float)
        if kind.lower() == "circle":
            out.append(("circle", *x[:3]))
        else:
            out.append(("hole" if neg else "polygon", x[0::2], x[1::2]))
    return out


def region_pixels(s_region, nside):
    """NESTED HEALPix pixels (at ``nside``) whose centres fall inside the footprint ``s_region``."""
    inside, holes = [np.array([], dtype=np.int64)], [np.array([], dtype=np.int64)]
    for shape in shapes(s_region):
        if shape[0] == "circle":
            _, ra, dec, r = shape
            inside.append(hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(r), nest=True))
        else:
            (holes if shape[0] == "hole" else inside).append(_polygon_pixels(nside, *shape[1:]))
    return np.setdiff1d(np.concatenate(inside), np.concatenate(holes))


def covered(regions, nside, buffer=2**26):
    """Unique pixels covered by any of ``regions`` and each region's pixel count; merged
    every ``buffer`` pixels so overlapping mosaics do not pile up in memory."""
    ids, pending, npix = np.array([], dtype=np.int64), [], {}
    for region in regions:
        pix = region_pixels(region, nside)
        npix[region] = len(pix)
        pending.append(pix)
        if sum(map(len, pending)) > buffer:
            ids, pending = np.unique(np.concatenate([ids, *pending])), []
    return np.unique(np.concatenate([ids, *pending])), npix


def _polygon_pixels(nside, ras, decs):
    """Pixels inside a (possibly non-convex) outline: candidates from its bounding disc,
    kept if inside the outline in the gnomonic projection about its centre."""
    vec = hp.ang2vec(ras, decs, lonlat=True)
    centre = vec.mean(axis=0)
    centre /= np.linalg.norm(centre)
    radius = np.arccos(np.clip(vec @ centre, -1, 1)).max()
    ids = hp.query_disc(nside, centre, radius, inclusive=True, nest=True)
    east = np.cross([0.0, 0.0, 1.0], centre)
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)

    def project(v):
        return np.column_stack([v @ east, v @ north]) / (v @ centre)[:, None]

    keep = MplPath(project(vec)).contains_points(project(np.column_stack(hp.pix2vec(nside, ids, nest=True))))
    return ids[keep]


# ---------------------------------------------------------------- plot


def plot(
    fields_path=FIELDS,
    out=Path("plots/alma_sky_coverage.png"),
    dec_max=30.0,
    nside=2**16,
    nside_map=32,
    extragalactic=False,
    zooms=ZOOMS,
):
    fields = Table.read(fields_path)
    fields = fields[fields["dec"] < dec_max]
    if extragalactic:
        fields = fields[np.isin(fields["category"], EXTRAGALACTIC)]
    pix_area = hp.nside2pixarea(nside, degrees=True) * 3600  # arcmin^2
    shift = 2 * (int(np.log2(nside)) - int(np.log2(nside_map)))

    bands = sorted(set(fields["band"]))
    union, stats = [], {}
    for band in bands:
        sel = fields[fields["band"] == band]
        ids, npix = covered(set(sel["s_region"]), nside)
        union.append(ids)
        area = np.array([npix[r] for r in sel["s_region"]]) * pix_area
        stats[band] = (len(sel), len(ids) * pix_area, float(np.sum(area * sel["bandwidth"])))
    ids = np.unique(np.concatenate(union))
    area_map = np.bincount(ids >> shift, minlength=hp.nside2npix(nside_map)) * pix_area

    fig = plt.figure(figsize=(14, 9.5), layout="constrained")
    grid = fig.add_gridspec(2, len(zooms) + 1, height_ratios=[1.35, 1])
    _sky(fig.add_subplot(grid[0, :-1], projection="mollweide"), area_map, nside_map, dec_max, zooms, fig)
    _table(fig.add_subplot(grid[0, -1]), stats, len(ids) * pix_area)
    for i, (name, centre) in enumerate(zooms.items()):
        _zoom(fig.add_subplot(grid[1, i]), fields, name, *centre)
    _band_legend(fig.add_subplot(grid[1, -1]), bands)
    what = "extragalactic projects" if extragalactic else "all science targets"
    fig.suptitle(f"ALMA interferometric footprints, Dec < {dec_max:+.0f} deg ({what}, {len(fields)} footprints)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"-> {out}")
    return fig


def _sky(ax, area_map, nside_map, dec_max, zooms, fig):
    n_ra, n_dec = 4 * nside_map * 2, 2 * nside_map * 2
    ra = np.linspace(-np.pi, np.pi, n_ra + 1)
    dec = np.linspace(-np.pi / 2, np.radians(dec_max), n_dec + 1)
    rc, dc = 0.5 * (ra[1:] + ra[:-1]), 0.5 * (dec[1:] + dec[:-1])
    R, D = np.meshgrid(rc, dc)
    # Mollweide x runs east to the left as on the sky: RA = -x
    cell = hp.ang2pix(nside_map, np.degrees(-R) % 360, np.degrees(D), lonlat=True, nest=True)
    values = np.where(area_map[cell] > 0, area_map[cell], np.nan)
    mesh = ax.pcolormesh(ra, dec, values, norm=LogNorm(vmin=0.1, vmax=max(np.nanmax(values), 1)), cmap="magma_r")
    fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02, label="covered area per cell [arcmin$^2$]")
    gal = SkyCoord(np.linspace(0, 360, 721), np.zeros(721), unit="deg", frame="galactic").icrs
    gx = -np.radians(np.where(gal.ra.deg > 180, gal.ra.deg - 360, gal.ra.deg))
    order = np.argsort(gx)
    gd = np.radians(gal.dec.deg)[order]
    ax.plot(gx[order], np.where(gd < np.radians(dec_max), gd, np.nan), color="0.5", lw=0.8, ls="--")
    for name, (ra0, dec0, _) in zooms.items():
        x = -np.radians(ra0 - 360 if ra0 > 180 else ra0)
        ax.annotate(name, (x, np.radians(dec0)), xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.plot(x, np.radians(dec0), "o", mfc="none", mec="C0", ms=6)
    ax.set_xticks(np.radians(np.arange(-150, 151, 30)))
    ax.set_xticklabels([f"{int(h)}h" for h in (np.arange(150, -151, -30) % 360) / 15])
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.fill_between(ra, np.radians(dec_max), np.pi / 2, color="0.88", lw=0, zorder=0)
    ax.set_title(f"HEALPix cells of {hp.nside2resol(nside_map, arcmin=True) / 60:.1f} deg", fontsize=10)


def _table(ax, stats, total):
    ax.axis("off")
    rows = [[f"{b}", f"{n}", f"{a:,.0f}", f"{abw:,.0f}"] for b, (n, a, abw) in stats.items()]
    rows.append(["all", f"{sum(s[0] for s in stats.values())}", f"{total:,.0f}", ""])
    table = ax.table(
        cellText=rows,
        colLabels=["band", "footprints", "area\n[arcmin$^2$]", "area x BW\n[arcmin$^2$ GHz]"],
        colWidths=[0.16, 0.24, 0.28, 0.32],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("unique area; area x bandwidth summed", fontsize=9)


def _zoom(ax, fields, name, ra0, dec0, half):
    centre = SkyCoord(ra0, dec0, unit="deg")
    near = fields[centre.separation(SkyCoord(fields["ra"], fields["dec"], unit="deg")).deg < 1.5 * half]
    near = near[np.argsort(-near["fov"])]  # big footprints under small ones
    for band in sorted(set(near["band"])):
        patches = []
        for region in near[near["band"] == band]["s_region"]:
            for shape in shapes(region):
                if shape[0] == "circle":
                    _, ra, dec, r = shape
                    x, y = _offset(centre, ra, dec)
                    patches.append(Circle((x[0], y[0]), r * 60))
                elif shape[0] == "polygon":
                    x, y = _offset(centre, shape[1], shape[2])
                    patches.append(Polygon(np.column_stack([x, y])))
        color = BAND_COLORS.get(band, "k")
        ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color, alpha=0.25, lw=0.3))
    ax.set_xlim(half * 60, -half * 60)
    ax.set_ylim(-half * 60, half * 60)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta\alpha$ [arcmin]")
    ax.set_ylabel(r"$\Delta\delta$ [arcmin]")
    ax.set_title(f"{name} ({len(near)} footprints)", fontsize=10)


def _offset(centre, ra, dec):
    dra, ddec = centre.spherical_offsets_to(SkyCoord(np.atleast_1d(ra), np.atleast_1d(dec), unit="deg"))
    return dra.arcmin, ddec.arcmin


def _band_legend(ax, bands):
    ax.axis("off")
    for band in bands:
        ax.fill([], [], color=BAND_COLORS.get(band, "k"), alpha=0.5, label=f"Band {band}")
    ax.legend(loc="center", frameon=False, title="half-power footprints")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("fetch", help="query the archive and write the footprint table")
    f.add_argument("--out", type=Path, default=FIELDS)
    f.add_argument("--tap", default="ESO", choices=["ESO", "NRAO", "NAOJ"])
    f.add_argument("--dec-max", type=float, default=30.0)
    f.add_argument("--public", action="store_true", help="public data only (default: all, positions are public)")
    p = sub.add_parser("plot", help="draw the coverage figure from the footprint table")
    p.add_argument("fields", type=Path, nargs="?", default=FIELDS)
    p.add_argument("--out", type=Path, default=Path("plots/alma_sky_coverage.png"))
    p.add_argument("--dec-max", type=float, default=30.0)
    p.add_argument("--nside", type=int, default=2**16, help="HEALPix nside of the area count (2^16: 3.2 arcsec)")
    p.add_argument("--nside-map", type=int, default=32, help="HEALPix nside of the sky map cells")
    p.add_argument("--extragalactic", action="store_true", help="only the extragalactic science categories")
    p.add_argument(
        "--zoom",
        action="append",
        metavar="NAME:RA:DEC:HALF",
        help="a zoom panel, centre and half-width in deg (repeatable; replaces the defaults)",
    )
    args = parser.parse_args()
    if args.step == "fetch":
        fetch(args.out, args.tap, args.dec_max, public=args.public)
    else:
        zooms = ZOOMS
        if args.zoom:
            zooms = {z.split(":")[0]: tuple(float(v) for v in z.split(":")[1:]) for z in args.zoom}
        plot(args.fields, args.out, args.dec_max, args.nside, args.nside_map, args.extragalactic, zooms)


if __name__ == "__main__":
    main()
