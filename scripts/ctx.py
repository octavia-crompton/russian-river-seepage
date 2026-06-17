import json
NB = "notebooks/2. Russian River  \u2013 analysis.ipynb"
with open(NB) as f:
    nb = json.load(f)
cells = nb["cells"]
for i in list(range(73, 80)) + list(range(107, 115)):
    c = cells[i]
    s = "".join(c.get("source", []))[:100].replace("\n", " ")
    print(f"cell {i+1}: id={c.get('id','?')}  src={s!r}")
