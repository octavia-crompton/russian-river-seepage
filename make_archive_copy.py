import json, copy

SRC = 'notebooks/2. Russian River  \u2013 analysis.ipynb'
DST = 'notebooks/archive/2. Russian River  \u2013 analysis 2026-02-27.ipynb'

with open(SRC) as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# ── lines to comment out: (cell_index_0based, line_index_0based) ──────────────
#  cell 47  lines 256,257  → plt.savefig(...)   contd.
#  cell 51  line  125      → pooled_event_df.to_csv(...)
#  cell 91  line  3        → summary_subset.to_csv('summary_for.csv')
#  cell 155 line  74       → shutil.copy2(src, dst)
TO_COMMENT = {
    46: [255, 256],   # savefig 2-liner
    50: [124],        # to_csv
    90: [2],          # to_csv
    154: [73],        # shutil.copy2
}

for ci, lines in TO_COMMENT.items():
    src = nb['cells'][ci]['source']
    for li in lines:
        if li < len(src) and not src[li].strip().startswith('#'):
            indent = len(src[li]) - len(src[li].lstrip())
            src[li] = src[li][:indent] + '# ' + src[li][indent:]
    nb['cells'][ci]['source'] = src

with open(DST, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved archive copy to:\n  {DST}")
print("Commented-out lines:")
for ci, lines in TO_COMMENT.items():
    for li in lines:
        print(f"  cell {ci+1} line {li+1}: {nb['cells'][ci]['source'][li].rstrip()}")
