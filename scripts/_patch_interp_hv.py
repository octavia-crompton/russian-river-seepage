"""Replace _predict_hv(xs) with actual h_V values in cell 69."""
import json, pathlib

nb_path = pathlib.Path(__file__).parent / "notebooks" / "2 Russian River  – analysis.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

# Find cell 69 by looking for the _predict_hv definition
target = None
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "_predict_hv" in src and "plot_two_models" in src:
        target = cell
        break

assert target is not None, "Could not find cell 69"

old_src = target["source"]
new_src = []
changes = 0

for line in old_src:
    # Per-event panels: replace global regression with actual h_V
    if "hs_reg   = _predict_hv(xs)" in line:
        line = line.replace(
            "# Smooth h_V curve: global linear regression h_V(Δh)",
            "# Smooth h_V curve: actual observed h_V at each Δh"
        ).replace(
            "hs_reg   = _predict_hv(xs)",
            "hs_reg   = hv[ord_]"
        )
        changes += 1
    # Pooled panel: replace global regression with actual h_V
    elif "hs_reg_p  = _predict_hv(xs_p)" in line:
        line = line.replace(
            "hs_reg_p  = _predict_hv(xs_p)",
            "hs_reg_p  = hv[ord_p]"
        )
        changes += 1
    new_src.append(line)

# Also fix the comment on the line before hs_reg_p if it exists
final_src = []
for line in new_src:
    if "# Smooth h_V curve: global linear regression h_V(Δh)" in line:
        line = line.replace(
            "# Smooth h_V curve: global linear regression h_V(Δh)",
            "# Smooth h_V curve: actual observed h_V at each Δh"
        )
    elif "smoothed via global linear regression" in line:
        line = line.replace(
            "smoothed via global linear regression h_V(Δh)",
            "actual observed h_V"
        )
    final_src.append(line)

target["source"] = final_src

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done – {changes} _predict_hv() calls replaced with actual h_V values")
