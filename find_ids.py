import json
NB = "notebooks/2. Russian River  \u2013 analysis.ipynb"
with open(NB) as f:
    nb = json.load(f)
cells = nb["cells"]

targets = [
    "len(summary)",
    "summary.head()",
    "summary['best_R2'].astype(float).median()",
    "summary['best_R2'].iloc[:23]",
    "summary['best_R2'].astype(float).mean()",
    "summary['b_best'].astype(float).mean()",
    "# --- driver code ---",
    "def get_row_USGS(",
    "def get_shading_regions(",
    "def plot_mass_bal_USGS(",
    "ind = 2",
    "def bootstrap_median_ci(",
]

for i, c in enumerate(cells):
    s = "".join(c.get("source", []))[:120].replace("\n", " ")
    cid = c.get("id", "?")
    for t in targets:
        if t in "".join(c.get("source", [])):
            print(f"  cell {i+1}: id={cid}  match={t!r}")
            print(f"    src={s!r}")
            break
