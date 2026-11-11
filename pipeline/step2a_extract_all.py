"""
Step 2a - Extract every VeReMi Extension archive (including the nested
per-hour zips) into one folder tree.

Safe to re-run at any time (for example while the download is still going):
  - archives that are incomplete / still downloading are skipped,
  - archives already extracted are skipped (a marker file is written),
  - inner zips are deleted after successful extraction to save disk space
    (use --keep-inner to keep them).

Usage:
    python step2a_extract_all.py --zips F:\\veremi_zips --out F:\\veremi
"""

import argparse
import os
import zipfile

MARKER = ".extracted_ok"


def extract_inner_zips(folder, keep_inner):
    """Extract every zip found under 'folder' (one level of nesting is typical)."""
    n = 0
    for dirpath, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".zip"):
                continue
            path = os.path.join(dirpath, name)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dirpath)
            n += 1
            if not keep_inner:
                os.remove(path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="Folder with the downloaded outer zips")
    ap.add_argument("--out", required=True, help="Destination folder")
    ap.add_argument("--keep-inner", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    outer = sorted(f for f in os.listdir(args.zips) if f.lower().endswith(".zip"))
    done, skipped, bad = 0, 0, []
    for name in outer:
        stem = name[:-4]
        target = os.path.join(args.out, stem)
        if os.path.exists(os.path.join(target, MARKER)):
            skipped += 1
            continue
        path = os.path.join(args.zips, name)
        try:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
            n_inner = extract_inner_zips(target, args.keep_inner)
            open(os.path.join(target, MARKER), "w").close()
            done += 1
            print(f"[OK]   {name}  (inner zips extracted: {n_inner})")
        except (zipfile.BadZipFile, EOFError, OSError) as e:
            bad.append(name)
            print(f"[SKIP] {name}  -> not a complete zip yet ({type(e).__name__})")

    print(f"\nExtracted now: {done} | already done: {skipped} | incomplete/skipped: {len(bad)}")
    if bad:
        print("Re-run this script after the download finishes to extract:", ", ".join(bad))


if __name__ == "__main__":
    main()
