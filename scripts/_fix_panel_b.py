import json, pathlib

nb_path = pathlib.Path("notebooks/2 Russian River  – analysis.ipynb")
nb = json.loads(nb_path.read_text())

target = "Pt Reyes h"
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if target in src and "USGS_filled" in src:
        old = src
        new = old.replace(
            "{'v' : 'Pt Reyes h', 'USGS_filled' : 'Estuary h', 'visitor_filled': 'visitor h'}",
            r"{'v' : r'Ocean $h_{\mathrm{ocn}}$', 'USGS_filled' : r'Estuary $h$'}",
        ).replace(
            '"m (NAVD)"',
            '"m (NAVD88)"',
        ).replace(
            "fontsize=18",
            "fontsize=FONTSIZE_PANEL",
        )
        if old == new:
            print("ERROR: no replacements matched")
            break
        # split back into lines for notebook JSON format
        lines = new.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        print(f"Cell {i}: patched successfully")
        break
else:
    print("ERROR: target cell not found")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("Saved.")
