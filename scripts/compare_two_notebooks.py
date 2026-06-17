#!/usr/bin/env python3
"""Compare two Jupyter notebooks in detail."""

import json
import sys

NB_A = "notebooks/2 Russian River  – analysis 3.ipynb"
NB_B = "notebooks/2 Russian River  – analysis.ipynb"

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cell_id(cell):
    return cell.get("id", None)

def cell_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

def summarize_cells(nb, label):
    cells = nb.get("cells", [])
    code = sum(1 for c in cells if c["cell_type"] == "code")
    md = sum(1 for c in cells if c["cell_type"] == "markdown")
    raw = sum(1 for c in cells if c["cell_type"] == "raw")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total cells : {len(cells)}")
    print(f"  Code cells  : {code}")
    print(f"  Markdown    : {md}")
    if raw:
        print(f"  Raw         : {raw}")

def trunc(s, n=120):
    s = s.replace("\n", "\\n")
    if len(s) > n:
        return s[:n] + "..."
    return s

def compare_metadata(nb_a, nb_b):
    print(f"\n{'='*60}")
    print("  METADATA / KERNELSPEC COMPARISON")
    print(f"{'='*60}")
    meta_a = nb_a.get("metadata", {})
    meta_b = nb_b.get("metadata", {})
    ks_a = meta_a.get("kernelspec", {})
    ks_b = meta_b.get("kernelspec", {})
    
    print(f"\n  nbformat: A={nb_a.get('nbformat')}.{nb_a.get('nbformat_minor')}  "
          f"B={nb_b.get('nbformat')}.{nb_b.get('nbformat_minor')}")
    
    print(f"\n  Kernelspec A: {json.dumps(ks_a, indent=2)}")
    print(f"  Kernelspec B: {json.dumps(ks_b, indent=2)}")
    if ks_a == ks_b:
        print("  -> Kernelspecs are IDENTICAL")
    else:
        print("  -> Kernelspecs DIFFER")

    lang_a = meta_a.get("language_info", {})
    lang_b = meta_b.get("language_info", {})
    if lang_a == lang_b:
        print("  -> language_info is IDENTICAL")
    else:
        print("  -> language_info DIFFERS")
        for key in sorted(set(list(lang_a.keys()) + list(lang_b.keys()))):
            va = lang_a.get(key, "<missing>")
            vb = lang_b.get(key, "<missing>")
            if va != vb:
                print(f"     {key}: A={va}  B={vb}")

def main():
    nb_a = load_nb(NB_A)
    nb_b = load_nb(NB_B)

    summarize_cells(nb_a, f"Notebook A: {NB_A}")
    summarize_cells(nb_b, f"Notebook B: {NB_B}")

    cells_a = nb_a["cells"]
    cells_b = nb_b["cells"]

    ids_a = [cell_id(c) for c in cells_a]
    ids_b = [cell_id(c) for c in cells_b]

    set_a = set(ids_a)
    set_b = set(ids_b)

    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    print(f"\n{'='*60}")
    print("  CELL ID COMPARISON")
    print(f"{'='*60}")
    print(f"  Shared cell IDs     : {len(shared)}")
    print(f"  Unique to A         : {len(only_a)}")
    print(f"  Unique to B         : {len(only_b)}")

    # Build lookup by id
    lookup_a = {cell_id(c): (i, c) for i, c in enumerate(cells_a)}
    lookup_b = {cell_id(c): (i, c) for i, c in enumerate(cells_b)}

    # Shared cells with differing source
    diffs = []
    same = 0
    for cid in shared:
        idx_a, ca = lookup_a[cid]
        idx_b, cb = lookup_b[cid]
        src_a = cell_source(ca)
        src_b = cell_source(cb)
        if src_a != src_b:
            diffs.append((cid, idx_a, idx_b, ca, cb, src_a, src_b))
        else:
            same += 1

    print(f"\n{'='*60}")
    print("  SHARED CELLS — SOURCE DIFFERENCES")
    print(f"{'='*60}")
    print(f"  Identical source    : {same}")
    print(f"  Different source    : {len(diffs)}")

    for cid, idx_a, idx_b, ca, cb, src_a, src_b in diffs:
        print(f"\n  Cell ID: {cid}  (A idx={idx_a}, B idx={idx_b})")
        print(f"    Type A: {ca['cell_type']}  Type B: {cb['cell_type']}")
        print(f"    A src: {trunc(src_a)}")
        print(f"    B src: {trunc(src_b)}")

    # Cells unique to A
    if only_a:
        print(f"\n{'='*60}")
        print("  CELLS UNIQUE TO A")
        print(f"{'='*60}")
        for cid in sorted(only_a, key=lambda x: lookup_a[x][0]):
            idx, c = lookup_a[cid]
            print(f"  idx={idx:3d}  type={c['cell_type']:8s}  id={cid}")
            print(f"    src: {trunc(cell_source(c))}")
    else:
        print("\n  No cells unique to A.")

    # Cells unique to B
    if only_b:
        print(f"\n{'='*60}")
        print("  CELLS UNIQUE TO B")
        print(f"{'='*60}")
        for cid in sorted(only_b, key=lambda x: lookup_b[x][0]):
            idx, c = lookup_b[cid]
            print(f"  idx={idx:3d}  type={c['cell_type']:8s}  id={cid}")
            print(f"    src: {trunc(cell_source(c))}")
    else:
        print("\n  No cells unique to B.")

    # Relative order of shared cells
    print(f"\n{'='*60}")
    print("  RELATIVE ORDER OF SHARED CELLS")
    print(f"{'='*60}")
    # Get shared cells in A order and B order
    shared_in_a_order = [cid for cid in ids_a if cid in shared]
    shared_in_b_order = [cid for cid in ids_b if cid in shared]

    if shared_in_a_order == shared_in_b_order:
        print("  -> Shared cells appear in the SAME relative order in both notebooks.")
    else:
        print("  -> Shared cells are in DIFFERENT relative order!")
        # Show first few order mismatches
        mismatches = 0
        for i, (a_id, b_id) in enumerate(zip(shared_in_a_order, shared_in_b_order)):
            if a_id != b_id:
                if mismatches < 15:
                    print(f"    Position {i}: A has {a_id}, B has {b_id}")
                mismatches += 1
        if mismatches > 15:
            print(f"    ... and {mismatches - 15} more order differences")
        if len(shared_in_a_order) != len(shared_in_b_order):
            print(f"    (lengths differ: A={len(shared_in_a_order)}, B={len(shared_in_b_order)})")

    # Compare metadata
    compare_metadata(nb_a, nb_b)

    # Check for output differences in shared cells
    print(f"\n{'='*60}")
    print("  SHARED CELLS — OUTPUT DIFFERENCES")
    print(f"{'='*60}")
    out_diffs = 0
    for cid in shared:
        _, ca = lookup_a[cid]
        _, cb = lookup_b[cid]
        oa = ca.get("outputs", [])
        ob = cb.get("outputs", [])
        if oa != ob:
            out_diffs += 1
    print(f"  Cells with different outputs: {out_diffs} / {len(shared)} shared cells")

    # Check execution counts
    print(f"\n{'='*60}")
    print("  SHARED CODE CELLS — EXECUTION COUNT DIFFERENCES")
    print(f"{'='*60}")
    exec_diffs = 0
    for cid in shared:
        _, ca = lookup_a[cid]
        _, cb = lookup_b[cid]
        if ca["cell_type"] == "code":
            ea = ca.get("execution_count")
            eb = cb.get("execution_count")
            if ea != eb:
                exec_diffs += 1
    shared_code = sum(1 for cid in shared if lookup_a[cid][1]["cell_type"] == "code")
    print(f"  Code cells with different exec counts: {exec_diffs} / {shared_code}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
