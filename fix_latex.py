import json

path = 'notebooks/2. Russian River  \u2013 analysis.ipynb'
with open(path) as f:
    nb = json.load(f)

fixed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'markdown':
        continue
    new_src = []
    for line in cell['source']:
        if '\\text{% error}' in line:
            new_line = line.replace('\\text{% error}', '\\text{\\% error}')
            print(f'Fixed: {repr(line)} -> {repr(new_line)}')
            new_src.append(new_line)
            fixed += 1
        else:
            new_src.append(line)
    cell['source'] = new_src

print(f'Total fixes: {fixed}')
if fixed:
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('Saved.')
