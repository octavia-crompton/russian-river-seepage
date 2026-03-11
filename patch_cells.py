import json, os

nb_path = "notebooks/2 Russian River  \u2013 analysis.ipynb"
nb = json.load(open(nb_path))

# Verify we have the right cells
assert "Summary statistics" in "".join(nb["cells"][68]["source"]), "Cell 68 mismatch"
assert "K_tilde_fmt" in "".join(nb["cells"][69]["source"]), "Cell 69 mismatch"

# Cell 68 – summary stats, updated to two-model notation
nb["cells"][68]["source"] = "\n".join([
    "# \u2500\u2500 Summary statistics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# Model 1 (linear):        S = C_L \u0394h",
    "# Model 2 (offset-power):  S = C_V \u03b6_V^{b_V} \u0394h",
    "print(f'N events: {len(summary)}')",
    "print()",
    "print('R\u00b2_L  (linear model,  S = C_L \u0394h):')",
    "print(f'  median R\u00b2_L :  {summary[\"R2\"].astype(float).median():.3f}  /  '",
    "      f'{summary[\"R2\"].iloc[:23].astype(float).median():.3f}  (all / first-23)')",
    "print(f'  mean   R\u00b2_L :  {summary[\"R2\"].astype(float).mean():.3f}')",
    "print()",
    "print('R\u00b2_V  (offset-power model,  S = C_V \u03b6_V^{{b_V}} \u0394h):')",
    "print(f'  median R\u00b2_V :  {summary[\"best_R2\"].astype(float).median():.3f}  /  '",
    "      f'{summary[\"best_R2\"].iloc[:23].astype(float).median():.3f}  (all / first-23)')",
    "print(f'  mean   R\u00b2_V :  {summary[\"best_R2\"].astype(float).mean():.3f}')",
    "print()",
    "print('b_V (offset-power exponent):')",
    "print(f'  mean (all events):  {summary[\"b_best\"].astype(float).mean():.3f}')",
    "print(f'  mean (first-20):    {summary[\"b_best\"].iloc[:20].astype(float).mean():.3f}')",
    "summary.head()",
])

# Cell 69 – LaTeX table, updated column header labels
nb["cells"][69]["source"] = "\n".join([
    "print(summary[['date', 'duration', 'N', 'K_tilde_fmt', 'R2', 'b_best', 'best_R2']].to_latex(index=False",
    "            ).replace('midrule', 'hline')",
    "             .replace('toprule', 'hline')",
    "             .replace('bottomrule', 'hline')",
    "             .replace('K_tilde_fmt', r'$C_L$')",
    "             .replace('b_best', r'$b_V$')",
    "             .replace('best_R2', r'$R^2_V$')",
    "             .replace('R2', r'$R^2_L$')",
    "             .replace('{lllllll}', '{lllll|ll}')",
    "      )",
])

json.dump(nb, open(nb_path, "w"), indent=1)
print("Done")
print("Cell 68:", "".join(nb["cells"][68]["source"]).splitlines()[0])
print("Cell 69:", "".join(nb["cells"][69]["source"]).splitlines()[0])
