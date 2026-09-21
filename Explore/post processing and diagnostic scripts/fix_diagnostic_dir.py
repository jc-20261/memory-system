#!/usr/bin/env python3
r"""
fix_diagnostic_paths.py – rewrite path constants in diagnostic scripts
that were moved into a subfolder of Explore.

For each .py file under TARGET_DIR:

    1. EXPLORE_DIR = Path(__file__).parent
       becomes
       EXPLORE_DIR = Path(__file__).resolve().parent.parent

       (with or without .resolve() in the original)

    2. MEMORIES_DIR = Path(r"F:\New folder (4)\New folder\Memories")
       becomes
       MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"

    3. Plain occurrences of the hardcoded path in comments or docstrings
       are left alone; only the assignments are rewritten.

Backs up each modified file as <name>.py.bak before writing.

Default is dry-run. Pass --apply to actually write.
"""

import argparse
import re
import shutil
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent   # the diagnostic subfolder

# Patterns
PAT_EXPLORE_DIR = re.compile(
    r'^(\s*)EXPLORE_DIR\s*=\s*Path\(__file__\)(?:\.resolve\(\))?\.parent\s*$',
    re.MULTILINE,
)
PAT_MEMORIES_DIR = re.compile(
    r'^(\s*)MEMORIES_DIR\s*=\s*Path\(\s*r?"F:\\New folder \(4\)\\New folder\\Memories"\s*\)\s*$',
    re.MULTILINE,
)

REPL_EXPLORE_DIR = r'\1EXPLORE_DIR = Path(__file__).resolve().parent.parent'
REPL_MEMORIES_DIR = r'\1MEMORIES_DIR = EXPLORE_DIR.parent / "Memories"'

SKIP_NAMES = {"fix_diagnostic_paths.py", "audit_dir.py", "apply_paths.py"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes (default is dry-run).")
    return p.parse_args()


def process_file(py: Path, apply: bool) -> int:
    """Return number of replacements in this file."""
    try:
        text = py.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [read-error] {py.name}: {e}")
        return 0

    new_text, n1 = PAT_EXPLORE_DIR.subn(REPL_EXPLORE_DIR, text)
    new_text, n2 = PAT_MEMORIES_DIR.subn(REPL_MEMORIES_DIR, new_text)

    total = n1 + n2
    if total == 0:
        return 0

    rel = py.relative_to(TARGET_DIR)
    tag = "[apply]" if apply else "[dry]  "
    print(f"  {tag} {rel}  (explore_dir={n1}, memories_dir={n2})")

    if apply:
        backup = py.with_suffix(py.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(py, backup)
        py.write_text(new_text, encoding="utf-8")

    return total


def main():
    args = parse_args()

    if not TARGET_DIR.exists():
        print(f"ERROR: target folder not found: {TARGET_DIR}")
        return

    print(f"Target: {TARGET_DIR}")
    print(f"Mode:   {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    total = 0
    files_changed = 0

    for py in sorted(TARGET_DIR.rglob("*.py")):
        if py.name in SKIP_NAMES:
            continue
        if "__pycache__" in py.parts:
            continue
        n = process_file(py, args.apply)
        if n:
            total += n
            files_changed += 1

    print()
    print("=" * 70)
    print(f"{'Applied' if args.apply else 'Would change'}: "
          f"{total} replacements across {files_changed} files")
    print("=" * 70)


if __name__ == "__main__":
    main()