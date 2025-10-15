#!/usr/bin/env python3
import argparse, os, re, sys
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", "build", "dist", ".venv", "venv", ".tox"}

# Matches lines that begin with "import" or "from" (ignoring leading spaces)
LINE_START = re.compile(r'^\s*(import|from)\b')
# Replace whole-word "PyParticle" (and dotted like PyParticle.something) on those lines
NAME = re.compile(r'\bPyParticle\b')

def should_skip_dir(d: str) -> bool:
    return d in SKIP_DIRS or d.startswith(".")

def rewrite_line(line: str) -> str:
    if not LINE_START.match(line):
        return line
    return NAME.sub("pyparticle", line)

def process_file(path: Path, apply_changes: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    changed = []
    out_lines = []
    for i, line in enumerate(text.splitlines(keepends=True), 1):
        nl = rewrite_line(line)
        if nl != line:
            changed.append(i)
        out_lines.append(nl)
    if not changed:
        return False
    rel = str(path)
    print(f"\n--- {rel}  ({len(changed)} change(s) on lines {changed[:6]}{'...' if len(changed)>6 else ''})")
    # unified diff preview (minimal)
    import difflib
    sys.stdout.writelines(difflib.unified_diff(text.splitlines(True), out_lines, fromfile=rel, tofile=rel))
    if apply_changes:
        # write a .bak once, then overwrite
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")
        path.write_text("".join(out_lines), encoding="utf-8")
    return True

def walk_repo(root: Path, apply_changes: bool) -> int:
    count = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not should_skip_dir(d)]
        for fn in fns:
            if fn.endswith(".py"):
                if process_file(Path(dp) / fn, apply_changes):
                    count += 1
    return count

def main():
    ap = argparse.ArgumentParser(description="Rewrite import/from PyParticle -> pyparticle (simple).")
    ap.add_argument("paths", nargs="+", help="Repo(s) or folders to scan")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = ap.parse_args()
    total = 0
    for p in args.paths:
        root = Path(p).resolve()
        if not root.exists():
            print(f"Path not found: {root}", file=sys.stderr); continue
        print(f"\nScanning {root} ...")
        total += walk_repo(root, args.apply)
    if total == 0:
        print("\nNo import lines needed changes.")
    else:
        print(f"\n{'Applied' if args.apply else 'Would change'} {total} file(s).")
        if not args.apply:
            print("Re-run with --apply to write changes. Backups (.bak) are created when writing.")
if __name__ == "__main__":
    main()
