import json

nb_path = "notebooks/2 Russian River  \u2013 analysis.ipynb"
nb = json.load(open(nb_path))

# Cell 69 – LaTeX table: add C_V, all power model params, ΔAIC and ΔR² for model comparison
# Columns: date | duration | N | C_L (K_tilde_fmt) | C_V (K_tilde_best) | b_V (b_best)
#        | R²_L (R2) | R²_V (best_R2) | ΔAIC (AIC_diff) | ΔR² (delta_R2)
nb["cells"][69]["source"] = "\n".join([
    "cols = ['date', 'duration', 'N', 'K_tilde_fmt', 'K_tilde_best', 'b_best',",
    "        'R2', 'best_R2', 'AIC_diff', 'delta_R2']",
    "print(summary[cols].to_latex(index=False)",
    "        .replace('midrule', 'hline')",
    "        .replace('toprule', 'hline')",
    "        .replace('bottomrule', 'hline')",
    "        .replace('K_tilde_fmt', r'$C_L$')",
    "        .replace('K_tilde_best', r'$C_V$')",
    "        .replace('b_best', r'$b_V$')",
    "        .replace('best_R2', r'$R^2_V$')",
    "        .replace('R2', r'$R^2_L$')",
    "        .replace('AIC_diff', r'$\\Delta$AIC')",
    "        .replace('delta_R2', r'$\\Delta R^2$')",
    "        .replace('{llllllllll}', '{llll|lll|lll}')",
    "      )",
])

json.dump(nb, open(nb_path, "w"), indent=1)
print("Done")
print(nb["cells"][69]["source"][:120])
