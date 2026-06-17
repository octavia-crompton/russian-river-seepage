# =====================================================================
# Figure 5 (panel layout): A-C = three LONGEST single events (per-event
# fits); D = pooled (all events). Companion to the single-panel version.
#
# Panels A-C use the offset-power (h_V) model with the SHAPE PARAMETERS
# FIXED at the pooled values (delta_V, b_V); only the conductance C_V is
# refit per event. The smooth h_V(delta h) curve is CLIPPED to each
# event's observed h_V range so the model line is not extrapolated below
# the data (which produced a spurious low-delta-h hook).
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pooled parameters (from c3) — used for panel D and as FIXED shape params in A-C
C_lin   = float(pooled_lin["C"])
C_vis   = float(pooled_vis["C"])
off_vis = float(pooled_vis["offset"])   # delta_V (fixed in panels A-C)
b_vis   = float(pooled_vis["b"])         # b_V    (fixed in panels A-C)

# pooled (centered) R2 on the full dataset
_x = df["delta_h"].to_numpy(float)
_y = df["seepage"].to_numpy(float)
_h = df["visitor_h"].to_numpy(float)
r2_lin_pool = r2_centered(_y, model_linear(C_lin, _x))
r2_vis_pool = r2_centered(_y, model_offset_power_visitor_h(C_vis, off_vis, b_vis, _x, _h, visitor_h_ref))

# global h_V ~ Δh regression (for the pooled panel's smooth visitor curve)
from numpy.polynomial.polynomial import polyfit as _polyfit, polyval as _polyval
_hv_coeffs = _polyfit(_x, _h, deg=1)

def _pooled_curves(x_dh):
    order = np.argsort(x_dh)
    xs = x_dh[order]
    hs = _polyval(xs, _hv_coeffs)
    yl  = model_linear(C_lin, xs)
    yvc = model_offset_power_visitor_h(C_vis, off_vis, b_vis, xs, hs, visitor_h_ref)
    return xs, yl, yvc

def _fit_CV_fixed_shape(x_dh, x_hv, y, off, b, m_ref):
    """Refit only the conductance C_V with delta_V (=off) and b_V (=b) fixed.
    The model S = C_V * delta_h * ((h_V+off)/m_ref)^b is linear in C_V, so the
    least-squares solution is closed form: C_V = <w, y> / <w, w>, w = delta_h*shape."""
    shape = np.power(np.clip((x_hv + off) / m_ref, 0.0, None), b)
    w = x_dh * shape
    denom = float(np.dot(w, w))
    C = float(np.dot(w, y) / denom) if denom > 0 else np.nan
    return max(C, 1e-12)

# --- three LONGEST closures (by duration in days), A = longest ---
_ev_dur = (df.assign(_d=pd.to_datetime(df["date"]))
             .groupby("event")
             .agg(dur=("_d", lambda s: (s.max() - s.min()).days),
                  n=("seepage", "size"))
             .sort_values(["dur", "n"], ascending=False))
EVENT_IDS = [int(e) for e in _ev_dur.index[:3]]

# canonical closure-start dates (match Table 1 / nb2 `summary`, which includes
# overwash days). nb3 uses the forecast-filtered subset, so its first data day
# can fall a few days after the true closure start; we label panels with the
# canonical start date so the figure is consistent with Table 1.
_t1 = pd.read_csv("../data/processed/summary_table1.csv")
_t1["start"] = pd.to_datetime(_t1["date"])
_t1["end"]   = _t1["start"] + pd.to_timedelta(_t1["duration"], unit="D")

def _canonical_start(sub):
    """Closure-start date from Table 1 whose [start, start+duration] window
    contains this event's median date; fall back to the nearest start."""
    d = pd.to_datetime(sub["date"])
    mid = d.median()
    hit = _t1[(_t1["start"] <= mid) & (mid <= _t1["end"])]
    if len(hit):
        return hit.iloc[0]["start"].date()
    j = (_t1["start"] - d.min()).abs().idxmin()
    return _t1.loc[j, "start"].date()

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
axd = axes.ravel()
panel_letters = ["A", "B", "C", "D"]

# panels A-C: single events with PER-EVENT linear fit + offset-power(h_V) fit
# (offset-power uses pooled delta_V, b_V; only C_V refit per event)
for k, ev in enumerate(EVENT_IDS):
    ax = axd[k]
    sub = df[df["event"] == ev]
    xv = sub["delta_h"].to_numpy(float)
    yv = sub["seepage"].to_numpy(float)
    hv = sub["visitor_h"].to_numpy(float)
    ax.scatter(xv, yv, s=20, c="0.5", alpha=0.7)   # no legend entry for data

    _ord = np.argsort(xv)
    xse = xv[_ord]

    # per-event linear fit (from grouped fits ev_lin)
    rl = ev_lin.loc[ev_lin["event"] == ev]
    C_L_ev  = float(rl["C_lin"].iloc[0])
    R2_L_ev = float(rl["R2_lin"].iloc[0])
    yl_ev = model_linear(C_L_ev, xse)
    lab_lin = rf"$S = {C_L_ev:.2f}\,\Delta h$  ($R^2={R2_L_ev:.2f}$)"
    ax.plot(xse, yl_ev, lw=2, c="C0", label=lab_lin)

    # per-event offset-power (h_V): delta_V, b_V FIXED at pooled; refit only C_V
    if np.isfinite(hv).sum() >= 3:
        C_V_ev = _fit_CV_fixed_shape(xv, hv, yv, off_vis, b_vis, visitor_h_ref)
        # R2 at the observed points
        yhat_pts = model_offset_power_visitor_h(C_V_ev, off_vis, b_vis, xv, hv, visitor_h_ref)
        R2_V_ev = r2_centered(yv, yhat_pts)
        # smooth h_V(Δh) within the event, CLIPPED to observed h_V range (no extrapolation)
        _c = np.polyfit(xv, hv, 1)
        hse = np.clip(np.polyval(_c, xse), float(np.nanmin(hv)), float(np.nanmax(hv)))
        yvc_ev = model_offset_power_visitor_h(C_V_ev, off_vis, b_vis, xse, hse, visitor_h_ref)
        _denom = visitor_h_ref + off_vis
        lab_vis = (rf"$S = {C_V_ev:.2f}\,\Delta h\left(\frac{{h_V{off_vis:+.2f}}}"
                   rf"{{{_denom:.2f}}}\right)^{{{b_vis:.2f}}}$  ($R^2={R2_V_ev:.2f}$)")
        ax.plot(xse, yvc_ev, lw=2, c="C1", label=lab_vis)

    try:
        dlab = _canonical_start(sub)
    except Exception:
        dlab = f"event {ev}"
    ax.set_title(f"{dlab}  (n = {len(sub)})", fontsize=11)
    ax.set_xlabel(r"$\Delta h$ (m)")
    ax.set_ylabel(r"$S$ (m$^3$ s$^{-1}$)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

# panel D: pooled (all events) with pooled fits
axp = axd[3]
axp.scatter(_x, _y, s=10, c="0.6", alpha=0.4)
_xs, _yl, _yvc = _pooled_curves(_x)
_denom_pool = visitor_h_ref + off_vis
lab_lin_pool = rf"$S = {C_lin:.2f}\,\Delta h$  ($R^2={r2_lin_pool:.2f}$)"
lab_vis_pool = (rf"$S = {C_vis:.2f}\,\Delta h\left(\frac{{h_V{off_vis:+.2f}}}"
                rf"{{{_denom_pool:.2f}}}\right)^{{{b_vis:.2f}}}$  ($R^2={r2_vis_pool:.2f}$)")
axp.plot(_xs, _yl,  lw=2, c="C0", label=lab_lin_pool)
axp.plot(_xs, _yvc, lw=2, c="C1", label=lab_vis_pool)
axp.set_title("Pooled (all events)", fontsize=11)
axp.set_xlabel(r"$\Delta h$ (m)")
axp.set_ylabel(r"$S$ (m$^3$ s$^{-1}$)")
axp.grid(alpha=0.2)
axp.legend(frameon=False, loc="upper left", fontsize=8)

# panel letters + tidy shared-axis labels
for k, ax in enumerate(axd):
    ax.text(-0.12, 1.08, panel_letters[k], transform=ax.transAxes,
            ha="left", va="top", fontsize=16, style="italic")
    ax.label_outer()

plt.tight_layout()
from pathlib import Path as _P
_figdir = _P("../figures"); _figdir.mkdir(exist_ok=True)
fig.savefig(_figdir / "fig5_model_comparison_panels.png", dpi=300, bbox_inches="tight")
plt.show()

print("Fig 5 panels: 3 longest events", EVENT_IDS,
      "durations", [int(_ev_dur.loc[e, "dur"]) for e in EVENT_IDS])
