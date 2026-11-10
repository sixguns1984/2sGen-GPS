#!/usr/bin/env python3
"""Conservative installer: add required Cairo backend to existing plot scripts.

It intentionally does NOT regex-rewrite unknown gene-label loops. Instead the
shared selector in common/ is integrated at the dataframe/label loop boundary
using the snippets in README. Full scripts must be inspected before changing
that boundary; silent global changes would risk panels b/c or evidence logic.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import sys

CAIRO = "import matplotlib\nmatplotlib.use('cairo')  # Use Cairo backend for high-quality output\n"


def get_targets(code_root: Path):
    root = code_root/'scripts_figures'
    out = []
    pd = root/'pd_figure2'/'01_make_pd_figure2.py'
    if pd.is_file(): out.append(pd)
    ad = root/'ad_figure4'
    if ad.is_dir():
        # All actual AD figure entrypoints importing pyplot are included rather
        # than guessing a non-verified AD script name.
        for f in sorted(ad.glob('*.py')):
            if 'matplotlib.pyplot' in f.read_text(encoding='utf-8'):
                out.append(f)
    if not pd.is_file():
        print('WARNING PD source not found:', pd)
    if not ad.is_dir():
        print('WARNING AD directory not found:', ad)
    return list(dict.fromkeys(out))


def patched_text(source: str, filename: Path):
    ast.parse(source)
    if "matplotlib.use('cairo')" in source or 'matplotlib.use("cairo")' in source:
        return source, 'already_has_cairo'
    lines = source.splitlines(keepends=True)
    pyplot_idx = next((i for i, line in enumerate(lines) if 'matplotlib.pyplot' in line and not line.lstrip().startswith('#')), None)
    if pyplot_idx is None:
        return source, 'no_pyplot_import'
    # Module import injection immediately before pyplot import avoids illegal
    # positioning ahead of shebang, module docstring or __future__ imports.
    lines.insert(pyplot_idx, CAIRO)
    result=''.join(lines)
    ast.parse(result)
    return result, 'add_cairo'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--code', required=True, type=Path, help='Original 2sGenGPS CODE directory')
    ap.add_argument('--apply', action='store_true', help='Without this: dry-run only')
    args=ap.parse_args()
    targets=get_targets(args.code)
    if not targets:
        raise SystemExit('No original drawing scripts found. Check CODE path and install patch files under CODE.')
    for f in targets:
        old=f.read_text(encoding='utf-8')
        new,status=patched_text(old,f)
        print(status, f)
        if args.apply and new!=old:
            backup=f.with_suffix(f.suffix+'.before_panel_a_patch.bak')
            if backup.exists():
                raise SystemExit('Refusing to overwrite existing backup '+str(backup))
            shutil.copy2(f,backup)
            f.write_text(new,encoding='utf-8')
            print('  backup:',backup)
    print('NOTE: label loop integration must use the original plotting dataframe. Read README.')

if __name__=='__main__': main()
