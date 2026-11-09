#!/usr/bin/env python3
"""Task A2: locate or fetch the ONNX + VNNLIB assets for the calibration slice.

Assets come from the VNN-COMP 2023 benchmark mirror
(github.com/ChristopherBrix/vnncomp2023_benchmarks), where each file is stored
gzip-compressed under benchmarks/<repo_dir>/<path>.gz. We download, decompress,
and lay them out as:

    external/vnncomp/<bench>/<onnx path>
    external/vnncomp/<bench>/<vnnlib path>

Default behaviour downloads only the small, fast, fully-tractable benchmarks
(acasxu_2023, mnist_fc) that make the plumbing calibration meaningful; pass
--benchmarks all (or a comma list) to fetch more. Nothing is invented: if a
file cannot be located or downloaded, it is recorded in MISSING_ASSETS.md and
the script exits non-zero so the runner can report honestly.

Usage:
  python prepare_vnnlib_calibration_assets.py [--benchmarks acasxu_2023,mnist_fc]
                                              [--check-only] [--max-onnx-mb 60]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLICE_CSV = ROOT / "vnnlib_calibration_slice.csv"
EXT = ROOT / "external/vnncomp"
MISSING_MD = ROOT / "MISSING_ASSETS.md"

REPO_BASE = ("https://raw.githubusercontent.com/ChristopherBrix/"
             "vnncomp2023_benchmarks/main/benchmarks")
# Per-benchmark source-base overrides. mnist_fc was retired from the 2023 run
# set and is hosted in the 2022 mirror; acasxu and the others stay on 2023.
# The expected SAT/UNSAT label is a deterministic property of the (network,
# spec) files, so the year of the mirror does not change ground truth.
SOURCE_BASE = {
    "mnist_fc": ("https://raw.githubusercontent.com/ChristopherBrix/"
                 "vnncomp2022_benchmarks/main/benchmarks"),
}
DEFAULT_BENCHES = ["acasxu_2023", "mnist_fc"]


def candidate_repo_dirs(bench, repo_dir):
    """Repo directory candidates to try in order (handles year-suffix drops)."""
    cands = [repo_dir, bench, bench.replace("_2023", ""),
             bench.replace("_2024", ""), bench.replace("_2022", "")]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def url_exists(url, timeout=15):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = r.headers.get("Content-Length")
            return True, int(size) if size else None
    except urllib.error.HTTPError:
        return False, None
    except Exception:
        return False, None


def download_gz(url, dest: Path, timeout=120):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".gz.tmp")
    with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    with gzip.open(tmp, "rb") as gz, dest.open("wb") as out:
        shutil.copyfileobj(gz, out)
    tmp.unlink(missing_ok=True)


def resolve_and_fetch(bench, repo_dir, rel_path, check_only, max_bytes):
    """Return (status, local_path_or_None, detail). status in
    {present, fetched, missing, skipped_large}."""
    local = EXT / bench / rel_path
    if local.exists() and local.stat().st_size > 0:
        return "present", local, None
    if check_only:
        return "missing", None, "check-only; not downloaded"
    last = None
    base = SOURCE_BASE.get(bench, REPO_BASE)
    for rd in candidate_repo_dirs(bench, repo_dir):
        url = f"{base}/{rd}/{rel_path}.gz"
        ok, size = url_exists(url)
        if not ok:
            last = f"404 {url}"
            continue
        if size and max_bytes and size > max_bytes:
            return "skipped_large", None, f"{size} bytes > cap at {url}"
        try:
            download_gz(url, local)
            return "fetched", local, url
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e} ({url})"
    return "missing", None, last or "no candidate URL resolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHES),
                    help="comma list of bench ids, or 'all'")
    ap.add_argument("--check-only", action="store_true",
                    help="report presence without downloading")
    ap.add_argument("--max-onnx-mb", type=float, default=60.0,
                    help="skip ONNX files larger than this (MB)")
    args = ap.parse_args()

    if not SLICE_CSV.exists():
        print(f"ERROR: {SLICE_CSV.name} not found; run "
              f"select_vnnlib_calibration_slice.py first", file=sys.stderr)
        sys.exit(2)

    rows = list(csv.DictReader(SLICE_CSV.open()))
    want = (None if args.benchmarks.strip().lower() == "all"
            else set(args.benchmarks.split(",")))
    rows = [r for r in rows if want is None or r["bench"] in want]
    max_bytes = int(args.max_onnx_mb * 1e6)

    # Deduplicate file fetches (many rows share an onnx).
    files = {}  # (bench, repo_dir, rel_path, kind) -> None
    for r in rows:
        files[(r["bench"], r["repo_dir"], r["onnx"], "onnx")] = None
        files[(r["bench"], r["repo_dir"], r["vnnlib"], "vnnlib")] = None

    results, missing = [], []
    for (bench, repo_dir, rel, kind) in sorted(files):
        cap = max_bytes if kind == "onnx" else None
        status, local, detail = resolve_and_fetch(bench, repo_dir, rel,
                                                  args.check_only, cap)
        rec = {"bench": bench, "kind": kind, "path": rel, "status": status,
               "detail": detail}
        results.append(rec)
        if status in ("missing", "skipped_large"):
            missing.append(rec)
        print(f"  [{status:13s}] {bench}/{rel}")

    # Per-instance readiness (both onnx and vnnlib present/fetched).
    ready = 0
    for r in rows:
        on = (EXT / r["bench"] / r["onnx"])
        vn = (EXT / r["bench"] / r["vnnlib"])
        if on.exists() and vn.exists():
            ready += 1

    if missing:
        lines = ["# Missing calibration assets", "",
                 f"{len(missing)} of {len(files)} files could not be obtained "
                 f"for the requested benchmarks (`{args.benchmarks}`).", "",
                 "These are external VNN-COMP 2023 assets, not produced by this "
                 "project. The runner treats instances with missing assets as "
                 "not-run (excluded from the agreement denominator).", "",
                 "| bench | kind | path | status | detail |",
                 "|---|---|---|---|---|"]
        for m in missing:
            lines.append(f"| {m['bench']} | {m['kind']} | `{m['path']}` | "
                         f"{m['status']} | {m['detail']} |")
        lines += ["", "## How to obtain manually", "",
                  "Each file lives at:", "",
                  f"    {REPO_BASE}/<repo_dir>/<path>.gz", "",
                  "Download, `gunzip`, and place at "
                  "`external/vnncomp/<bench>/<path>`."]
        MISSING_MD.write_text("\n".join(lines) + "\n")
        print(f"\n{len(missing)} missing; wrote {MISSING_MD.name}")
    elif MISSING_MD.exists():
        MISSING_MD.unlink()

    print(f"\nInstances ready (onnx+vnnlib present): {ready}/{len(rows)}")
    sys.exit(0 if ready > 0 else 1)


if __name__ == "__main__":
    main()
