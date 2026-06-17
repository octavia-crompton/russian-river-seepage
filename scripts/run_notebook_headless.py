#!/usr/bin/env python
"""
Headless notebook executor for the Russian River seepage pipeline.

Runs a notebook on the active (hydro) Jupyter kernel via nbclient, executing
cells top-to-bottom. A lone top-level ``break`` statement (an interactive manual
stop-point in nb2) is neutralised before execution so it does not raise a
SyntaxError. Execution continues past errors, but EVERY error cell is reported
afterwards with its traceback so nothing is masked.

Usage:
    python run_notebook_headless.py "<notebook.ipynb>" [--save]
"""
import sys
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    nb_path = Path(args[0]).resolve()
    save = "--save" in flags

    nb = nbformat.read(nb_path, as_version=4)

    # Neutralise any lone top-level `break` cell (invalid outside a loop).
    neutralised = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if cell.source.strip() == "break":
            cell.source = "pass  # skipped: invalid top-level `break` (manual stop-point)"
            neutralised.append(i)
    if neutralised:
        print(f"[info] neutralised lone break cell(s) at index {neutralised}")

    kernel_name = next((a.split("=", 1)[1] for a in flags if a.startswith("--kernel=")), "hydro")
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name=kernel_name,
        allow_errors=True,  # continue past errors; we report them all below
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    print(f"[info] executing {nb_path.name} ({len(nb.cells)} cells) ...")

    # Live per-cell progress hooks.
    _t = {"start": None}

    def _on_start(cell, cell_index=None, **kw):
        _t["start"] = time.time()

    def _on_done(cell, cell_index=None, execute_reply=None, **kw):
        dt = time.time() - (_t["start"] or time.time())
        ec = cell.get("execution_count")
        first = (cell.get("source", "").strip().splitlines() or [""])[0][:60]
        err = ""
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                err = f"  !! {o.get('ename')}: {str(o.get('evalue'))[:80]}"
        print(f"  [{cell_index:>3}] ec={ec} {dt:6.1f}s | {first}{err}", flush=True)

    client.on_cell_start = _on_start
    client.on_cell_executed = _on_done

    client.execute()

    if save:
        nbformat.write(nb, nb_path)
        print(f"[info] saved executed notebook -> {nb_path.name}")

    # Report every error cell transparently.
    errors = []
    code_idx = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code_idx += 1
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                tb = out.get("traceback", [])
                tb_clean = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(tb))
                first = (cell.source.strip().splitlines() or [""])[0]
                errors.append((i, code_idx, out.get("ename"), out.get("evalue"),
                               first, tb_clean))

    print("\n" + "=" * 70)
    if not errors:
        print("[OK] notebook executed with NO error outputs.")
    else:
        print(f"[ERRORS] {len(errors)} cell(s) produced error outputs:")
        for i, ci, ename, evalue, first, tb_clean in errors:
            print(f"\n--- cell index {i} (code cell #{ci}) ---")
            print(f"    src : {first}")
            print(f"    err : {ename}: {evalue}")
            tail = tb_clean.strip().splitlines()[-4:]
            for line in tail:
                print(f"    | {line}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
