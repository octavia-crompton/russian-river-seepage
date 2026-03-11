import json

nb_path = "notebooks/2 Russian River  \u2013 analysis.ipynb"
nb = json.load(open(nb_path))

# Cell 68 – summary stats, full two-model notation
nb["cells"][68]["source"] = "\n".join([
    "# \u2500\u2500 Summary statistics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    "# Model 1 (linear):        S = C_L * \u0394h                 -> summary['R2'], summary['K_tilde']",
    "# Model 2 (power on \u0394h):   S = C_V * \u0394h^{b_V}           -> summary['best_R2'], summary['K_tilde_best'], summary['b_best']",
    "print(f'N events: {len(summary)}')",
    "print()",
    "print('--- Linear model  (S = C_L \u0394h) ---')",
    "print(f'  C_L : median = {summary[\"K_tilde\"].astype(float).median():.3f}')",
    "print(f'  R\u00b2_L : median = {summary[\"R2\"].astype(float).median():.3f}  /  '",
    "       f'{summary[\"R2\"].iloc[:23].astype(float).median():.3f}  (all / first-23)')",
    "print()",
    "print('--- Power model  (S = C_V \u0394h^{{b_V}}) ---')",
    "print(f'  C_V : median = {summary[\"K_tilde_best\"].astype(float).median():.3f}')",
    "print(f'  b_V : mean (all)     = {summary[\"b_best\"].astype(float).mean():.3f}')",
    "print(f'  b_V : mean (first-20) = {summary[\"b_best\"].iloc[:20].astype(float).mean():.3f}')",
    "print(f'  R\u00b2_V : median = {summary[\"best_R2\"].astype(float).median():.3f}  /  '",
    "       f'{summary[\"best_R2\"].iloc[:23].astype(float).median():.3f}  (all / first-23)')",
    "print()",
    "print('--- Model comparison ---')",
    "print(f'  \u0394R\u00b2 = R\u00b2_V - R\u00b2_L : median = {summary[\"delta_R2\"].astype(float).median():.3f}')",
    "print(f'  \u0394AIC = AIC_L - AIC_V : median = {summary[\"AIC_diff\"].astype(float).median():.1f}  (positive \u21d2 power model better)')",
    "print(f'  N events where power model preferred (\u0394AIC > 2): '",
    "       f'{(summary[\"AIC_diff\"].astype(float) > 2).sum()} / {len(summary)}')",
    "summary.head()",
])

json.dump(nb, open(nb_path, "w"), indent=1)
print("Done")
