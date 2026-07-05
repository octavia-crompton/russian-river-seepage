"""
Russian River estuary - Figure 1 (two-panel location map)
==========================================================
Pulls imagery from Google Earth Engine, reprojects to UTM Zone 10N
(EPSG:32610), and draws a regional context panel + a high-res detail
panel with scale bars, a north arrow, a detail-footprint box with
connector lines, and your hand-drawn feature lines -- all in matplotlib.

Detail panel  : NAIP aerial (~0.6 m, USDA public domain)
Regional panel: Sentinel-2 (10 m, Copernicus open data)  [default]
                ...or NAIP downsampled, for the tightest colour match
                (see USE_NAIP_FOR_REGIONAL below).

Both datasets keep their own open licence -- Earth Engine is only the
pipe, so the figure is fine to publish under CC BY. Credit the data
sources ("USDA NAIP", "Copernicus Sentinel-2") and acknowledge GEE.

One-time setup
--------------
    pip install earthengine-api geemap rasterio numpy matplotlib \
                matplotlib-scalebar pyproj geedim
    earthengine authenticate          # only needed once

Then edit the seven CONFIG values below and run:  python russian_river_fig1.py
"""

import os
import numpy as np
import ee
import geemap
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer

# ============================================================
# CONFIG  -- the only things you should need to change
# ============================================================
PROJECT   = "ee-quatratavia"      # your registered Cloud project
OUT_DIR   = "./fig1_out"              # where GeoTIFFs + figure are written

# Detail panel: centre on the river mouth + half-width (metres)
DETAIL_CENTER = (-123.1280, 38.4500)  # (lon, lat) -- shifted ~250 m EAST of the
                                       # mouth so the estuary fills the frame and
                                       # the glinty open ocean mostly drops out.
                                       # Nudge lon east (less negative) for more
                                       # estuary, west (more negative) for more ocean.
DETAIL_HALF_M = 350                    # -> ~700 m wide detail panel

# Regional panel: bounding box [west, south, east, north] in lon/lat
# (default spans the mouth down past Pt. Reyes to San Francisco Bay)
REGION_BBOX = [-123.45, 37.70, -122.35, 38.60]

NAIP_YEAR   = 2022                           # SINGLE NAIP year (one flight = no
                                             # mosaic seam). CA options incl.
                                             # 2018, 2020, 2022 -- try a couple
                                             # and keep whichever has least glint.
S2_WINDOW   = ("2022-06-01", "2022-09-30")   # dry-season Sentinel-2 window
UTM_CRS     = "EPSG:32610"                    # UTM 10N -> axes are metres, north up

REGION_SCALE_M = 40             # regional pixel size (m). 40 keeps the request
                                # small and looks fine at panel size; lower is
                                # sharper + a bigger download (auto-tiles if it
                                # exceeds Earth Engine's ~48 MB direct limit).

USE_NAIP_FOR_REGIONAL = False   # True = NAIP (downsampled) for BOTH panels
                                # -> tightest colour match, larger download

FLAT_OCEAN_REGIONAL = True      # paint the regional ocean a flat colour. The
                                # compositing seams live almost entirely over
                                # water, so this removes them cleanly. Only the
                                # regional panel -- the detail panel keeps real
                                # water so the estuary/lagoon stays visible.
OCEAN_RGB = (38, 66, 104)       # flat ocean colour (R, G, B)

FLAT_OCEAN_DETAIL = False       # leave the detail panel as real imagery (the
                                # glint/seam is handled by recentring east, not
                                # by masking). Flattening it kept causing trouble;
                                # if you ever want to try again it now uses the
                                # fixed coastline mask, not the old blocky one.
OCEAN_EAST_LON = None           # None = flatten ALL water (ocean + estuary),
                                # matching panel A. To keep the estuary as real
                                # imagery, set a longitude in the dry berm
                                # between ocean and estuary (e.g. -123.128) and
                                # only water west of it is flattened.

DRAW_FEATURES = False           # labelled feature lines on the detail panel
NORTH_ARROW_DETAIL = False      # north arrow on the detail panel
DEBUG_WATER_MASK = False        # also write detail_watermask.tif (cyan = water,
                                # dark = land) so you can see what's flattened
SCALE_FONTSIZE = 15             # scale-bar label font size (both panels)
PANEL_LABEL_FONTSIZE = 18       # A / B panel-letter font size

# Feature lines to draw on the DETAIL panel. Digitise points in Google
# Earth (right-click -> "what's here" or add a path) and paste lon/lat
# here. Each feature is a label + a list of (lon, lat) vertices.
FEATURES = {
    "Northern limit of migration": [(-123.1318, 38.4523), (-123.1300, 38.4521)],
    "Jetty / southern limit":      [(-123.1312, 38.4476), (-123.1296, 38.4478)],
    "Rock escarpment":             [(-123.1307, 38.4505), (-123.1293, 38.4503)],
}
POINTS = {                       # single labelled points
    "Observer's vantage pt.": (-123.1289, 38.4498),
}
# ============================================================


def naip_window(year):
    return (f"{year}-01-01", f"{year}-12-31")


def mask_s2_clouds(img):
    """Drop cloud-shadow / cloud / cirrus pixels using the SCL band, so the
    median composite isn't pulled around by haze (a big source of seams)."""
    scl = img.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(bad.Not())


def _ocean_image():
    return (ee.Image.constant(list(OCEAN_RGB))
            .rename(["vis-red", "vis-green", "vis-blue"]).uint8())


def flatten_ndwi(vis, source, green, nir):
    """Flatten water (NDWI > 0) to a solid colour -- used for the regional
    panel, where there's no strong glint to fool the index."""
    water = source.normalizedDifference([green, nir]).gt(0.0)
    return vis.where(water, _ocean_image())


def detail_water_mask(geom):
    """NDWI>0 water mask from a multi-year Sentinel-2 composite over the detail
    area -- built exactly like the regional mask that works. The coast is foggy
    in summer, so the multi-year window is what guarantees full coverage. No
    resample/unmask: those collapsed the mask to a single coarse cell, which is
    what flooded the whole panel with ocean colour."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom)
          .filterDate("2021-01-01", "2022-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .map(mask_s2_clouds).median())
    return s2.normalizedDifference(["B3", "B8"]).gt(0.0)


def export_image(image, filename, scale, region, crs=UTM_CRS):
    """Download an Earth Engine image to a GeoTIFF, robustly.

    Tries Earth Engine's fast direct-download endpoint first. That endpoint
    caps a single request at ~48 MB and -- annoyingly -- prints the size
    error instead of raising it, so we check whether a file actually landed
    and, if not, fall back to a tiled download (via geedim) that has no size
    limit. The detail panel is tiny and always takes the fast path; only a
    large regional pull would ever need the fallback.
    """
    try:
        geemap.ee_export_image(image, filename=filename,
                               scale=scale, region=region, crs=crs)
    except Exception as e:                       # noqa: BLE001
        print(f"  direct download raised: {e}")
    if os.path.exists(filename):
        return filename

    print("  request too large for the direct endpoint; tiling instead...")
    try:
        geemap.download_ee_image(image, filename,
                                 region=region, crs=crs, scale=scale)
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(
            f"Could not download {filename}. The tiled fallback needs geedim "
            "(`pip install geedim`); otherwise raise REGION_SCALE_M to shrink "
            "the request."
        ) from e
    if not os.path.exists(filename):
        raise RuntimeError(f"Download produced no file: {filename}")
    return filename


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ee.Initialize(project=PROJECT)

    # --- geometries -------------------------------------------------
    detail_geom = (ee.Geometry.Point(DETAIL_CENTER)
                   .buffer(DETAIL_HALF_M).bounds())
    region_geom = ee.Geometry.Rectangle(REGION_BBOX)

    # --- detail image: NAIP true colour, 8-bit (single year) -------
    naip = (ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(detail_geom)
            .filterDate(*naip_window(NAIP_YEAR))
            .mosaic())
    detail_vis = naip.select(["R", "G", "B"]).visualize(min=0, max=255)
    if FLAT_OCEAN_DETAIL:
        water = detail_water_mask(detail_geom)
        if OCEAN_EAST_LON is not None:
            lon = ee.Image.pixelLonLat().select("longitude")
            water = water.And(lon.lt(OCEAN_EAST_LON))
        detail_vis = detail_vis.where(water, _ocean_image())
        if DEBUG_WATER_MASK:
            export_image(
                water.visualize(min=0, max=1, palette=["221122", "00e5ff"]),
                os.path.join(OUT_DIR, "detail_watermask.tif"),
                scale=2, region=detail_geom)

    detail_tif = os.path.join(OUT_DIR, "detail_naip.tif")
    export_image(detail_vis, detail_tif, scale=0.6, region=detail_geom)

    # --- regional image -------------------------------------------
    region_tif = os.path.join(OUT_DIR, "region.tif")
    if USE_NAIP_FOR_REGIONAL:
        reg = (ee.ImageCollection("USDA/NAIP/DOQQ")
               .filterBounds(region_geom).filterDate(*naip_window(NAIP_YEAR))
               .mosaic())
        region_vis = reg.select(["R", "G", "B"]).visualize(min=0, max=255)
        if FLAT_OCEAN_REGIONAL:
            region_vis = flatten_ndwi(region_vis, reg, "G", "N")
    else:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region_geom).filterDate(*S2_WINDOW)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
              .map(mask_s2_clouds)
              .median())
        region_vis = s2.visualize(bands=["B4", "B3", "B2"],
                                  min=0, max=3000, gamma=1.2)
        if FLAT_OCEAN_REGIONAL:
            region_vis = flatten_ndwi(region_vis, s2, "B3", "B8")

    export_image(region_vis, region_tif, scale=REGION_SCALE_M, region=region_geom)

    build_figure(region_tif, detail_tif, detail_geom)


# ------------------------------------------------------------------
# Matplotlib figure
# ------------------------------------------------------------------
def _read(path):
    """Return (rgb array, [left,right,bottom,top]) in the file's CRS."""
    with rasterio.open(path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        b = src.bounds
    return rgb, [b.left, b.right, b.bottom, b.top]


def _to_utm(lon, lat):
    t = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    return t.transform(lon, lat)


def _stroke(c="white"):
    return [pe.withStroke(linewidth=2.5, foreground="black")] if c == "white" \
        else [pe.withStroke(linewidth=2.5, foreground="white")]


def _north_arrow(ax):
    ax.annotate("", xy=(0.93, 0.90), xytext=(0.93, 0.80),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2, color="white"))
    ax.text(0.93, 0.93, "N", transform=ax.transAxes, ha="center",
            color="white", fontweight="bold", fontsize=15,
            # path_effects=_stroke("white")
            )


def build_figure(region_tif, detail_tif, detail_geom):
    reg_rgb, reg_ext = _read(region_tif)
    det_rgb, det_ext = _read(detail_tif)

    # Make both panels the SAME height and top-aligned (so A and B line up too).
    # width_ratios set to each panel's data aspect (width/height) lets equal-
    # aspect (undistorted) imagery come out the same height; anchor "N" tops them.
    asp_r = (reg_ext[1] - reg_ext[0]) / (reg_ext[3] - reg_ext[2])
    asp_d = (det_ext[1] - det_ext[0]) / (det_ext[3] - det_ext[2])
    fig, (axr, axd) = plt.subplots(1, 2, figsize=(11, 6),
                                   gridspec_kw={"width_ratios": [asp_r, asp_d]})
    axr.set_anchor("N")
    axd.set_anchor("N")

    # --- regional panel -------------------------------------------
    axr.imshow(reg_rgb, extent=reg_ext, origin="upper")
    axr.set_xlim(reg_ext[0], reg_ext[1]); axr.set_ylim(reg_ext[2], reg_ext[3])
    axr.set_xticks([]); axr.set_yticks([])
    axr.add_artist(ScaleBar(1, units="m", location="lower left",
                            color="white", box_alpha=0, scale_loc="top",
                            font_properties={"size": SCALE_FONTSIZE,
                                             "weight": "bold"}))
    _north_arrow(axr)

    # detail footprint box on the regional panel
    dl, dr, db, dt = det_ext
    axr.add_patch(Rectangle((dl, db), dr - dl, dt - db, fill=False,
                            edgecolor="yellow", lw=1.5))

    # --- detail panel ---------------------------------------------
    axd.imshow(det_rgb, extent=det_ext, origin="upper")
    axd.set_xlim(det_ext[0], det_ext[1]); axd.set_ylim(det_ext[2], det_ext[3])
    axd.set_xticks([]); axd.set_yticks([])
    axd.add_artist(ScaleBar(1, units="m", location="lower left",
                            color="white", box_alpha=0, scale_loc="top",
                            font_properties={"size": SCALE_FONTSIZE,
                                             "weight": "bold"}))
    if NORTH_ARROW_DETAIL:
        _north_arrow(axd)

    # feature lines + labels (lon/lat -> UTM); clip_on keeps a mis-digitised
    # point from drawing its label across into the regional panel
    if DRAW_FEATURES:
        for label, verts in FEATURES.items():
            xy = [_to_utm(lon, lat) for lon, lat in verts]
            xs, ys = zip(*xy)
            axd.plot(xs, ys, color="white", lw=2, clip_on=True,
                     path_effects=_stroke("white"))
            axd.annotate(label, xy[0], color="white", fontsize=8, clip_on=True,
                         path_effects=_stroke("white"))
        for label, (lon, lat) in POINTS.items():
            x, y = _to_utm(lon, lat)
            axd.plot(x, y, marker="^", color="white", ms=8, clip_on=True,
                     path_effects=_stroke("white"))
            axd.annotate(label, (x, y), color="white", fontsize=8, clip_on=True,
                         xytext=(6, 6), textcoords="offset points",
                         path_effects=_stroke("white"))

    # connector lines from the box (regional) to the detail panel corners
    for (xa, ya), (xb, yb) in [((dr, dt), (det_ext[0], det_ext[3])),
                               ((dr, db), (det_ext[0], det_ext[2]))]:
        fig.add_artist(ConnectionPatch(
            xyA=(xa, ya), coordsA=axr.transData,
            xyB=(xb, yb), coordsB=axd.transData,
            color="yellow", lw=0.8))

    # panel letters: italic, above each panel's top-left corner (matches Fig 5)
    for ax, letter in ((axr, "A"), (axd, "B")):
        ax.text(0.0, 1.01, letter, transform=ax.transAxes,
                fontstyle="italic", fontsize=PANEL_LABEL_FONTSIZE,
                va="bottom", ha="left")

    fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.94, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"figure1.{ext}"),
                    dpi=400, bbox_inches="tight")
    print("wrote figure1.pdf / figure1.png to", OUT_DIR)


if __name__ == "__main__":
    main()