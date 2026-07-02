#!/usr/bin/env python
"""Rewrite the shared base-path prefix in a generated configurations.xlsx.

The generated configs build ``folder``, ``data_file`` and ``run_file_mcmc`` from a
common prefix (default ``Species360_Calibration``). When moving from the initial
calibration run to a real Baysian03 run, repoint that prefix in one shot::

    ~/SRvenv/bin/python app/set_base_prefix.py configurations_auto.xlsx --new-prefix Baysian03

Only those three path rows change; theta and timing are untouched, and all other
sheets (e.g. QC) are preserved.
"""

import argparse
import sys

import pandas as pd

#: Rows whose values carry the base prefix.
PATH_ROWS = ("folder", "data_file", "run_file_mcmc")


def _repoint(value, old_prefix, new_prefix):
    if not isinstance(value, str):
        return value
    if value == old_prefix:
        return new_prefix
    if value.startswith(old_prefix + "/"):
        return new_prefix + value[len(old_prefix):]
    return value


def rewrite_prefix(in_path, new_prefix, old_prefix="Species360_Calibration", out_path=None):
    """Rewrite the base prefix in ``PATH_ROWS`` of Sheet1; preserve other sheets.

    Returns the number of cells changed.
    """
    out_path = out_path or in_path
    sheets = pd.read_excel(in_path, sheet_name=None, index_col=0, engine="openpyxl")
    if "Sheet1" not in sheets:
        # fall back to the first sheet (ExcelConfigParser treats it as DEFAULT)
        first = next(iter(sheets))
        sheets = {("Sheet1" if k == first else k): v for k, v in sheets.items()}

    df = sheets["Sheet1"]
    changed = 0
    for row in PATH_ROWS:
        if row not in df.index:
            continue
        for col in df.columns:
            old = df.at[row, col]
            new = _repoint(old, old_prefix, new_prefix)
            if new != old:
                df.at[row, col] = new
                changed += 1

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            # QC sheet has a default RangeIndex; write it without the index column.
            write_index = (name == "Sheet1")
            sdf.to_excel(writer, sheet_name=name, index=write_index, header=True)
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("excel", help="path to a generated configurations .xlsx")
    parser.add_argument("--new-prefix", required=True, help="new base prefix")
    parser.add_argument("--old-prefix", default="Species360_Calibration",
                        help="prefix to replace (default: the initial-run prefix)")
    parser.add_argument("--out", default=None, help="output path (default: in place)")
    args = parser.parse_args(argv)

    changed = rewrite_prefix(args.excel, args.new_prefix,
                             old_prefix=args.old_prefix, out_path=args.out)
    dest = args.out or args.excel
    print(f"Rewrote {changed} path cell(s): {args.old_prefix!r} -> {args.new_prefix!r} in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
