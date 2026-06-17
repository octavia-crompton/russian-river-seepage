import json, pathlib

nb_path = pathlib.Path("notebooks/2 Russian River  – analysis.ipynb")
nb = json.loads(nb_path.read_text())

target = "def plot_visitor_scatter"
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if target in src:
        old = src
        new = old.replace(
            r'label = r"All days; $S + O = C_L\,\Delta h$"',
            r'label = r"All days; $S + O$"',
        ).replace(
            r'label = r"Non-overwash days; $S = C_L\,\Delta h$"',
            r'label = r"Non-overwash days; $S$"',
        )
        if old == new:
            print("ERROR: no replacements matched")
            break
        lines = new.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        print(f"Cell {i}: patched successfully")
        break
else:
    print("ERROR: target cell not found")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("Saved.")
