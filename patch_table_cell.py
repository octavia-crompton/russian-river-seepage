import json

nb_path = '/Users/octaviacrompton/Google_Drive_quatratavia/estuaries_jenner/russian-river-seepage/notebooks/2 Russian River  \u2013 analysis.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "cols = ['date', 'duration', 'N', 'K_tilde_best', 'b_best',\n",
    "        'R2', 'best_R2', 'AIC_diff', 'delta_R2']\n",
    "numeric_cols = ['K_tilde_best', 'b_best', 'R2', 'best_R2', 'AIC_diff', 'delta_R2']\n",
    "tbl = summary[cols].copy()\n",
    "for c in numeric_cols:\n",
    "    tbl[c] = tbl[c].astype(float).round(2)\n",
    "latex_str = tbl.to_latex(index=False)\n",
    "# Replace column names with LaTeX (order matters: specific before general)\n",
    "col_renames = {\n",
    "    'K_tilde_best': r'$C_V$',\n",
    "    'b_best':       r'$b_V$',\n",
    "    'best_R2':      r'$R^2_V$',\n",
    "    'delta_R2':     r'$\\Delta R^2$',\n",
    "    'AIC_diff':     r'$\\Delta$AIC',\n",
    "    'R2':           r'$R^2_L$',\n",
    "}\n",
    "for old, new in col_renames.items():\n",
    "    latex_str = latex_str.replace(old, new)\n",
    "latex_str = (latex_str\n",
    "    .replace('midrule', 'hline')\n",
    "    .replace('toprule', 'hline')\n",
    "    .replace('bottomrule', 'hline')\n",
    "    .replace('{lllllllll}', '{lll|lll|lll}')\n",
    ")\n",
    "print(latex_str)"
]

target_id = '87340d92'
changed = False
for cell in nb['cells']:
    if cell.get('id') == target_id:
        cell['source'] = new_source
        cell['outputs'] = []
        cell['execution_count'] = None
        changed = True
        print(f'Patched cell {target_id}')
        break

if not changed:
    print('ERROR: cell not found')
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('File written successfully.')
