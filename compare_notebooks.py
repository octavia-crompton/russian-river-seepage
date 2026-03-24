#!/usr/bin/env python3
"""Compare two Jupyter notebooks cell by cell."""

import json
import sys

NB_A = "notebooks/2 Russian River  – analysis keep.ipynb"
NB_B = "notebooks/2 Russian River  – analysis.ipynb"

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cell_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

def cell_id(cell):
    return cell.get("id", None)

def summarize(nb, label):
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    raw_cells = [c for c in cells if c["cell_type"] == "raw"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total cells: {len(cells)}")
    print(f"  Code cells:  {len(code_cells)}")
    print(f"  Markdown:    {len(md_cells)}")
    if raw_cells:
        print(f"  Raw:         {len(raw_cells)}")
    print(f"\n  {'Idx':>4}  {'Type':>8}  {'ID':>40}  First 100 chars of source")
    print(f"  {'-'*4}  {'-'*8}  {'-'*40}  {'-'*50}")
    for i, c in enumerate(cells):
        src = cell_source(c).replace("\n", "\\n")[:100]
        cid = cell_id(c) or "(no id)"
        print(f"  {i:4d}  {c['cell_type']:>8}  {cid:>40}  {src}")
    return cells

def compare_metadata(nb_a, nb_b):
    print(f"\n{'='*70}")
    print("  METADATA COMPARISON")
    print(f"{'='*70}")
    # kernel
    ks_a = nb_a.get("metadata", {}).get("kernelspec", {})
    ks_b = nb_b.get("metadata", {}).get("kernelspec", {})
    if ks_a != ks_b:
        print(f"  Kernelspec differs:")
        print(f"    A: {ks_a}")
        print(f"    B: {ks_b}")
    else:
        print(f"  Kernelspec: same ({ks_a.get('display_name', '?')})")

    li_a = nb_a.get("metadata", {}).get("language_info", {})
    li_b = nb_b.get("metadata", {}).get("language_info", {})
    if li_a != li_b:
        print(f"  language_info differs:")
        print(f"    A: {li_a}")
        print(f"    B: {li_b}")
    else:
        print(f"  language_info: same")

    fmt_a = nb_a.get("nbformat"), nb_a.get("nbformat_minor")
    fmt_b = nb_b.get("nbformat"), nb_b.get("nbformat_minor")
    if fmt_a != fmt_b:
        print(f"  nbformat differs: A={fmt_a}, B={fmt_b}")
    else:
        print(f"  nbformat: same ({fmt_a[0]}.{fmt_a[1]})")

def compare_cells(cells_a, cells_b):
    print(f"\n{'='*70}")
    print("  CELL-BY-CELL COMPARISON")
    print(f"{'='*70}")

    ids_a = [cell_id(c) for c in cells_a]
    ids_b = [cell_id(c) for c in cells_b]
    set_a = set(ids_a)
    set_b = set(ids_b)

    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    print(f"\n  Cell IDs in A: {len(ids_a)}")
    print(f"  Cell IDs in B: {len(ids_b)}")
    print(f"  Shared IDs:    {len(shared)}")
    print(f"  Only in A:     {len(only_a)}")
    print(f"  Only in B:     {len(only_b)}")

    # Build lookup by ID
    lookup_a = {}
    for i, c in enumerate(cells_a):
        cid = cell_id(c)
        if cid:
            lookup_a[cid] = (i, c)
    lookup_b = {}
    for i, c in enumerate(cells_b):
        cid = cell_id(c)
        if cid:
            lookup_b[cid] = (i, c)

    # Cells only in A
    if only_a:
        print(f"\n  --- Cells ONLY in A (keep) ---")
        for cid in sorted(only_a, key=lambda x: lookup_a[x][0]):
            idx, c = lookup_a[cid]
            src = cell_source(c).replace("\n", "\\n")[:120]
            print(f"    idx={idx:3d}  type={c['cell_type']:8s}  id={cid}  src: {src}")

    # Cells only in B
    if only_b:
        print(f"\n  --- Cells ONLY in B (analysis) ---")
        for cid in sorted(only_b, key=lambda x: lookup_b[x][0]):
            idx, c = lookup_b[cid]
            src = cell_source(c).replace("\n", "\\n")[:120]
            print(f"    idx={idx:3d}  type={c['cell_type']:8s}  id={cid}  src: {src}")

    # Shared cells - check for differences
    diffs = []
    same = 0
    for cid in shared:
        idx_a, c_a = lookup_a[cid]
        idx_b, c_b = lookup_b[cid]
        src_a = cell_source(c_a)
        src_b = cell_source(c_b)
        type_a = c_a["cell_type"]
        type_b = c_b["cell_type"]
        if src_a != src_b or type_a != type_b:
            diffs.append((cid, idx_a, idx_b, c_a, c_b, src_a, src_b))
        else:
            same += 1

    print(f"\n  Shared cells with IDENTICAL source: {same}")
    print(f"  Shared cells with DIFFERENT source: {len(diffs)}")

    if diffs:
        print(f"\n  --- Shared cells with DIFFERENT content ---")
        for cid, idx_a, idx_b, c_a, c_b, src_a, src_b in sorted(diffs, key=lambda x: x[1]):
            print(f"\n    ID: {cid}")
            print(f"    A idx={idx_a}, B idx={idx_b}")
            print(f"    A type={c_a['cell_type']}, B type={c_b['cell_type']}")
            print(f"    A len={len(src_a)}, B len={len(src_b)}")
            # Show first 150 chars of each
            sa = src_a.replace("\n", "\\n")[:150]
            sb = src_b.replace("\n", "\\n")[:150]
            print(f"    A: {sa}")
            print(f"    B: {sb}")
            # Find first difference position
            minlen = min(len(src_a), len(src_b))
            diff_pos = None
            for p in range(minlen):
                if src_a[p] != src_b[p]:
                    diff_pos = p
                    break
            if diff_pos is None and len(src_a) != len(src_b):
                diff_pos = minlen
            if diff_pos is not None:
                ctx_start = max(0, diff_pos - 20)
                ctx_end_a = min(len(src_a), diff_pos + 40)
                ctx_end_b = min(len(src_b), diff_pos + 40)
                print(f"    First diff at char {diff_pos}:")
                print(f"      A[{ctx_start}:{ctx_end_a}]: {repr(src_a[ctx_start:ctx_end_a])}")
                print(f"      B[{ctx_start}:{ctx_end_b}]: {repr(src_b[ctx_start:ctx_end_b])}")

    # Check ordering
    print(f"\n  --- ORDER COMPARISON ---")
    shared_order_a = [cid for cid in ids_a if cid in shared]
    shared_order_b = [cid for cid in ids_b if cid in shared]
    if shared_order_a == shared_order_b:
        print("  Shared cells appear in the SAME order in both notebooks.")
    else:
        print("  Shared cells appear in DIFFERENT order!")
        # Find first difference in ordering
        for i in range(min(len(shared_order_a), len(shared_order_b))):
            if shared_order_a[i] != shared_order_b[i]:
                print(f"    First order difference at shared-index {i}:")
                print(f"      A: {shared_order_a[i]} (A idx={lookup_a[shared_order_a[i]][0]})")
                print(f"      B: {shared_order_b[i]} (B idx={lookup_b[shared_order_b[i]][0]})")
                break

def main():
    nb_a = load_nb(NB_A)
    nb_b = load_nb(NB_B)

    cells_a = summarize(nb_a, f"A: {NB_A}")
    cells_b = summarize(nb_b, f"B: {NB_B}")
    compare_metadata(nb_a, nb_b)
    compare_cells(cells_a, cells_b)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
