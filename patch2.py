import json

path = "/Users/octaviacrompton/Google_Drive_quatratavia/estuaries_jenner/russian-river-seepage/notebooks/2 Russian River  \u2013 analysis.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

new_source = (
    "from scipy.signal import savgol_filter\n"
    "\n"
    "# Smooth V(h) with a Savitzky-Golay filter (window=21, poly=3)\n"
    'vol_smooth = savgol_filter(hypso["vol_m3"].values, window_length=21, polyorder=3)\n'
    'hypso["vol_smooth_m3"] = vol_smooth\n'
    "\n"
    "# Recompute derivatives from the smoothed volume\n"
    'dVdh_smooth = np.gradient(vol_smooth, hypso["h_m"].values)\n'
    'hypso["dVdh_m2"] = dVdh_smooth\n'
    "\n"
    'd2Vdh2_smooth = np.gradient(dVdh_smooth, hypso["h_m"].values)\n'
    'hypso["d2Vdh2_m"] = d2Vdh2_smooth\n'
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)\n"
    "\n"
    "# (a) V(h): raw + smoothed\n"
    "ax = axes[0]\n"
    'ax.plot(hypso["h_m"], hypso["vol_m3"], color="lightgrey", linewidth=1, label="raw")\n'
    'ax.plot(hypso["h_m"], vol_smooth, linewidth=1.5, label="smoothed")\n'
    'ax.set_ylabel("Volume V(h) (m$^3$)")\n'
    'ax.set_xlabel("$h$ (m NAVD)")\n'
    'ax.set_title("Hypsometric curve V(h)")\n'
    "ax.legend(frameon=False, fontsize=8)\n"
    "\n"
    "# (b) First derivative dV/dh (from smoothed V)\n"
    "ax = axes[1]\n"
    'ax.plot(hypso["h_m"], hypso["dVdh_m2"])\n'
    'ax.set_ylabel(r"$\\mathrm{d}V/\\mathrm{d}h$ (m$^2$)")\n'
    'ax.set_xlabel("$h$ (m NAVD)")\n'
    'ax.set_title("Slope A(h) = dV/dh  [smoothed V]")\n'
    "\n"
    "# (c) Second derivative d\u00b2V/dh\u00b2 (from smoothed V)\n"
    "ax = axes[2]\n"
    'ax.plot(hypso["h_m"], hypso["d2Vdh2_m"])\n'
    'ax.axhline(0, color="grey", linewidth=0.8)\n'
    'ax.set_ylabel(r"$\\mathrm{d}^2 V/\\mathrm{d} h^2$ (m)")\n'
    'ax.set_xlabel("$h$ (m NAVD)")\n'
    'ax.set_title("Curvature  [smoothed V]")\n'
    "\n"
    "ax.set_xlim(-1, 5)\n"
    "plt.tight_layout()"
)

old_marker = "# Assuming hypso already has h_m and vol_m3 and we've computed dVdh above"
seen = 0
patched = False
for cell in nb["cells"]:
    joined = "".join(cell.get("source", []))
    if old_marker in joined:
        seen += 1
        if seen == 2:  # target the second duplicate (cell 13)
            cell["source"] = new_source
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Patched cell id={cell.get('id')}")
            patched = True
            break

if not patched and seen == 1:
    for cell in nb["cells"]:
        if old_marker in "".join(cell.get("source", [])):
            cell["source"] = new_source
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Patched (only) cell id={cell.get('id')}")
            patched = True
            break

if not patched:
    print(f"ERROR: not patched (seen={seen})")
else:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Saved.")
