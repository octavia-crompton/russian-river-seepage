"""
Refactor notebook:
  1. Consolidate scratchpad one-liners (cells 67-76)
  2. Split 3 large monolithic cells at natural section boundaries
"""
import json, uuid, re

NB = "notebooks/2. Russian River  \u2013 analysis.ipynb"

with open(NB) as f:
    nb = json.load(f)

cells = nb["cells"]

def src(c):
    return "".join(c.get("source", []))

def find(cell_id):
    for i, c in enumerate(cells):
        if c.get("id", "") == cell_id:
            return i
    raise KeyError(f"cell id {cell_id!r} not found")

def new_code(source_str, uid=None):
    return {
        "cell_type": "code",
        "id": uid or uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_str,
        "outputs": [],
        "execution_count": None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scratchpad one-liners
# ─────────────────────────────────────────────────────────────────────────────
DELETE_IDS = [
    "3eba6bd4",  # len(summary)  [before travel time]
    "0c477fdb",  # summary.head() [before travel time]
    "f00d00c5",  # subset [:23] median
    "9e1d55ea",  # mean R2
    "6080b38e",  # b_best mean
    "61830542",  # summary.head() [duplicate after latex]
    "d4c6e059",  # len(summary[...]) [after latex]
]

STATS_ID = "c08eaea2"
CONSOLIDATED_STATS = (
    "# \u2500\u2500 Summary statistics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "print(f\"N events: {len(summary)}\")\n"
    "print(f\"\\nR\u00b2 (all events / first-23 events):\")\n"
    "print(f\"  median R\u00b2:        {summary['R2'].astype(float).median():.3f}  /  \"\n"
    "      f\"{summary['R2'].iloc[:23].astype(float).median():.3f}\")\n"
    "print(f\"  median R\u00b2_optim:  {summary['best_R2'].astype(float).median():.3f}  /  \"\n"
    "      f\"{summary['best_R2'].iloc[:23].astype(float).median():.3f}\")\n"
    "print(f\"  mean   R\u00b2:        {summary['R2'].astype(float).mean():.3f}\")\n"
    "print(f\"  mean   R\u00b2_optim:  {summary['best_R2'].astype(float).mean():.3f}\")\n"
    "print(f\"\\nb_best: mean (all)={summary['b_best'].astype(float).mean():.3f}  \"\n"
    "      f\"mean (first-20)={summary['b_best'].iloc[:20].astype(float).mean():.3f}\")\n"
    "summary.head()"
)

# Apply deletions
cells = [c for c in cells if c.get("id", "") not in DELETE_IDS]
nb["cells"] = cells

# Edit consolidated stats cell
idx = find(STATS_ID)
cells[idx]["source"] = CONSOLIDATED_STATS

print("Scratchpad consolidation done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Split cell 48: VSC-e7a22e1e  (visitor plot functions + driver loop)
# ─────────────────────────────────────────────────────────────────────────────
C48_ID = "6da785fd"
idx48 = find(C48_ID)
full48 = src(cells[idx48])

SPLIT_MARKER_48 = "# --- driver code ---"
if SPLIT_MARKER_48 in full48:
    part_a, part_b = full48.split(SPLIT_MARKER_48, 1)
    cells[idx48]["source"] = part_a.rstrip("\n")
    driver_cell = new_code(
        "# \u2500\u2500 Driver: plot one closure event per iteration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        + part_b.lstrip("\n")
    )
    cells.insert(idx48 + 1, driver_cell)
    print("Cell 48 split OK")
else:
    print("WARNING: split marker not found in cell 48")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Split cell 84: VSC-e8b4ecdf  (visitor fns | USGS fns)
# ─────────────────────────────────────────────────────────────────────────────
C84_ID = "3f768843"
idx84 = find(C84_ID)
full84 = src(cells[idx84])

SPLIT_MARKER_84 = "\ndef get_row_USGS("
if SPLIT_MARKER_84 in full84:
    pos = full84.index(SPLIT_MARKER_84)
    part_a = full84[:pos].rstrip("\n")
    part_b = full84[pos].lstrip("\n") + full84[pos+1:]
    # part_b starts from \ndef get_row_USGS — strip leading newline
    part_b = full84[pos:].lstrip("\n")
    cells[idx84]["source"] = part_a
    usgs_cell = new_code(
        "# \u2500\u2500 USGS pipeline helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        + part_b
    )
    cells.insert(idx84 + 1, usgs_cell)
    print("Cell 84 split OK")
else:
    print("WARNING: split marker not found in cell 84 — trying alternate...")
    # Try matching def get_row_USGS at start of line
    m = re.search(r'\ndef get_row_USGS\b', full84)
    if m:
        pos = m.start()
        part_a = full84[:pos].rstrip("\n")
        part_b = full84[pos:].lstrip("\n")
        cells[idx84]["source"] = part_a
        usgs_cell = new_code(
            "# \u2500\u2500 USGS pipeline helpers \u2500\u2500\u2500\u2500\n"
            + part_b
        )
        cells.insert(idx84 + 1, usgs_cell)
        print("Cell 84 split OK (alternate)")
    else:
        print("ERROR: Could not split cell 84")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Split cell 111: 3da51188  (USGS plot fns | driver)
# Cell 109 (utility fns) already clean; cell 111 has plot fns + driver mixed
# ─────────────────────────────────────────────────────────────────────────────
C111_ID = "3da51188"
idx111 = find(C111_ID)
full111 = src(cells[idx111])

# Split before the driver code block (starts with 'ind = 2')
m2 = re.search(r'\nind = 2\b', full111)
if m2:
    p2_start = m2.start()
    plot_part   = full111[:p2_start].rstrip("\n")
    driver_part = full111[p2_start:].lstrip("\n")

    cells[idx111]["source"] = plot_part
    driver_cell_111 = new_code(
        "# \u2500\u2500 Driver: plot one USGS closure event \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        + driver_part
    )
    cells.insert(idx111 + 1, driver_cell_111)
    print("Cell 111 split OK (plot fns / driver)")
else:
    print("WARNING: 'ind = 2' not found in cell 111 — checking with different pattern...")
    m2b = re.search(r'\nind\s*=\s*2\b', full111)
    if m2b:
        p2_start = m2b.start()
        plot_part   = full111[:p2_start].rstrip("\n")
        driver_part = full111[p2_start:].lstrip("\n")
        cells[idx111]["source"] = plot_part
        driver_cell_111 = new_code(
            "# \u2500\u2500 Driver: plot one USGS closure event \u2500\u2500\n" + driver_part
        )
        cells.insert(idx111 + 1, driver_cell_111)
        print("Cell 111 split OK (alternate match)")
    else:
        print("ERROR: could not split cell 111")

# ─────────────────────────────────────────────────────────────────────────────
# Write back
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone. Total cells: {len(cells)}")
