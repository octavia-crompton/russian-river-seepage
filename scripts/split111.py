import json, re, uuid

NB = "notebooks/2. Russian River  \u2013 analysis.ipynb"
with open(NB) as f:
    nb = json.load(f)
cells = nb["cells"]

for i, c in enumerate(cells):
    if c.get("id") == "3da51188":
        s = "".join(c.get("source", []))
        # driver starts at '\nind = '
        m = re.search(r'\nind\s*=\s*\d+\b', s)
        if m:
            pos = m.start()
            fn_part  = s[:pos].rstrip("\n")
            drv_part = s[pos:].lstrip("\n")
            c["source"] = fn_part
            drv_cell = {
                "cell_type": "code",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": (
                    "# \u2500\u2500 Driver: plot one USGS closure event "
                    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
                    + drv_part
                ),
                "outputs": [],
                "execution_count": None,
            }
            cells.insert(i + 1, drv_cell)
            print(f"Split cell 111 at pos {pos}: fn_part={len(fn_part)} chars, drv_part={len(drv_part)} chars")
        else:
            print("ERROR: could not find driver code boundary")
        break

nb["cells"] = cells
with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Total cells: {len(cells)}")
