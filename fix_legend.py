import json, os

files = os.listdir('notebooks')
nb_file = [f for f in files if 'analysis' in f and not f.startswith('2.')][0]
path = 'notebooks/' + nb_file
with open(path) as f:
    nb = json.load(f)
cell_map = {c['id']: c for c in nb['cells']}

def patch(cell, old, new, label):
    src = ''.join(cell['source'])
    if old not in src:
        print(f'  NOT FOUND [{label}]')
        return
    new_src = src.replace(old, new)
    lines = new_src.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print(f'  OK [{label}]')

c = cell_map['3af0813a']

# plot_mass_bal_USGS – replace if/else legend block (has ax.set_xlabel after it)
patch(c,
    "    if legend_loc == 'outside':\n"
    "        ax.legend(title = '', loc='center left', bbox_to_anchor=(1, 0.5))\n"
    "    else:\n"
    "        ax.legend(title = '', loc=legend_loc)\n"
    "    ax.set_xlabel('')",
    "    ax.legend(title='', loc='upper left', fontsize=10)\n"
    "    ax.set_xlabel('')",
    '104-mass_bal legend')

# plot_12hr_USGS – replace if/else legend block (has blank line + set_ylim after it)
patch(c,
    "    if legend_loc == 'outside':\n"
    "        ax.legend(title = '', loc='center left', bbox_to_anchor=(1, 0.5))\n"
    "    else:\n"
    "        ax.legend(title = '', loc=legend_loc)\n"
    "\n"
    "    ax.set_ylim(ax.get_ylim()[0]*0.5, ax.get_ylim()[1])",
    "    ax.legend(title='', loc='upper left', fontsize=10)\n"
    "\n"
    "    ax.set_ylim(ax.get_ylim()[0]*0.5, ax.get_ylim()[1])",
    '104-12hr legend')

# plot_USGS_scatter
patch(c,
    "    ax.legend(loc = 'upper left')\n"
    "    ax.set_xlim(0, )",
    "    ax.legend(loc='upper left', fontsize=10)\n"
    "    ax.set_xlim(0, )",
    '104-scatter legend')

c2 = cell_map['3da51188']

# plot_mass_bal_USGS redefined in cell 106
patch(c2,
    "    ax.legend(title = '', loc='center left', bbox_to_anchor=(1, 0.5))\n"
    "    ax.set_xlabel('')",
    "    ax.legend(title='', loc='upper left', fontsize=10)\n"
    "    ax.set_xlabel('')",
    '106-mass_bal legend')

# plot_USGS_scatter redefined in cell 106
patch(c2,
    "    ax.legend(loc = 'upper left')\n"
    "    ax.set_xlim(pos_subset['delta_h'].min()*0.75, )",
    "    ax.legend(loc='upper left', fontsize=10)\n"
    "    ax.set_xlim(pos_subset['delta_h'].min()*0.75, )",
    '106-scatter legend')

with open(path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
