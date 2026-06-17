import json, glob

for f in sorted(glob.glob("notebooks/**/*.ipynb", recursive=True)):
    with open(f) as fh:
        nb = json.load(fh)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "Pt Reyes" in src and "Estuary" in src:
            s = src[:1500]
            print(f"=== {f} Cell {i} ===")
            print(s)
            print("---")
