import json

path = 'notebooks/2. Russian River  \u2013 analysis.ipynb'
with open(path) as f:
    nb = json.load(f)

# ── 1. Delete duplicate cell 93 ───────────────────────────────────────────────
# ── 3. Consolidate scratchpad cells 52-60: delete superseded/one-liners ───────
#    Keep: 54 (IQR threshold), 56 (scatter plot)
#    Merge 57-60 one-liners into a single "stats" cell after the plot
#    Delete: 52 (mean+3std, superseded), 53 (display), 55 (display), 57-60
#    Merge 57-60 content into one new cell
# ── 5. Delete cell 97 (dead plot_with_delta_h function) ──────────────────────
DELETE_IDS = {
    'd254c59b',   # 1: duplicate R2 cell (cell 93)
    '4c6da8cc',   # 3: superseded max_threshold (cell 52)
    'b75d5080',   # 3: display of superseded threshold (cell 53)
    'a641f88a',   # 3: display of threshold (cell 55)
    'e8cef2b1',   # 3: one-liner (cell 57)
    '8fdceed0',   # 3: one-liner (cell 58)
    '9deb428a',   # 3: one-liner (cell 59)
    '7df48bb3',   # 3: one-liner (cell 60)
    '2a5a8880',   # 5: dead plot_with_delta_h function (cell 97)
}

# The merged stats cell will be inserted after cell 56 (id 43caf215)
AFTER_ID = '43caf215'
MERGED_STATS_CELL = {
    "cell_type": "code",
    "source": [
        "# exploratory stats\n",
        "print('max_threshold (IQR):', max_threshold)\n",
        "print('seepage/delta_h std:', (pooled_event_df['seepage']/pooled_event_df['delta_h']).std())\n",
        "print('3*std, mean:', 3*pooled_event_df.seepage.std(), pooled_event_df.seepage.mean())\n",
        "print('seepage max:', pooled_event_df.seepage.max())\n",
        "print('shape:', pooled_event_df.shape)\n",
    ],
    "metadata": {},
    "outputs": [],
    "execution_count": None
}

# ── 2. Fix merget_no_wave typo ────────────────────────────────────────────────
# ── 4. Fix to_csv path in cell 91 ────────────────────────────────────────────
EDIT_MAP = {
    # cell 21 assignment: merget_no_wave → merged_no_wave
    # find by id
}

new_cells = []
for cell in nb['cells']:
    cid = cell.get('id', '')

    # ── 2. fix typo ──
    if any('merget_no_wave' in l for l in cell['source']):
        cell['source'] = [l.replace('merget_no_wave', 'merged_no_wave') for l in cell['source']]
        print(f"  Fixed merget_no_wave typo in cell id={cid}")

    # ── 4. fix to_csv path ──
    if cid == '2539801e':
        cell['source'] = [
            l.replace("summary_subset.to_csv('summary_for.csv')",
                      "summary_subset.to_csv(data_path + 'processed/summary_for.csv')")
            for l in cell['source']
        ]
        print(f"  Fixed to_csv path in cell id={cid}")

    if cid in DELETE_IDS:
        print(f"  Deleted cell id={cid}")
        continue

    new_cells.append(cell)

    # ── 3. insert merged stats cell after scatter plot cell ──
    if cid == AFTER_ID:
        new_cells.append(MERGED_STATS_CELL)
        print(f"  Inserted merged stats cell after id={cid}")

nb['cells'] = new_cells
print(f"\nTotal cells: {len(nb['cells'])}")

with open(path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved.")
