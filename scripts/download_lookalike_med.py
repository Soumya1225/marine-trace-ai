"""Download the Eastern Mediterranean oil / look-alike dataset (Yang & Singha 2025, PANGAEA.980773, CC-BY-4.0).

Reads the file list Datasets/lookalike_med/PANGAEA_980773.tab and downloads every image (.jpg) and label (.xml)
into Datasets/lookalike_med/<set>/ where set is ow / oc (oil, water / coast) or nw / nc (no oil = look-alikes).
Files already downloaded are skipped, so the command can be run again after an interruption.

    python scripts/download_lookalike_med.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "Datasets" / "lookalike_med"
TAB = DIR / "PANGAEA_980773.tab"
URL = "https://download.pangaea.de/dataset/980773/files/{}"


def read_tab(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if l.strip() == "*/"), 0)
    files = set()
    for line in lines[start + 1:]:                      # skip the column header
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        subset = cols[0].strip()
        for name in (cols[1].strip(), cols[2].strip()):
            if name and subset:
                files.add((subset, name))
    return sorted(files)


def fetch(subset, name, session):
    out = DIR / subset / name
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = session.get(URL.format(name), timeout=60)
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            tmp = out.with_suffix(out.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(out)
            return "ok"
        except Exception:                                # noqa: BLE001 - retry on network errors
            time.sleep(2 * (attempt + 1))
    return "failed"


def main():
    if not TAB.exists():
        sys.exit(f"File list not found: {TAB}")
    files = read_tab(TAB)
    print(f"{len(files)} files listed ({sum(f[1].endswith('.jpg') for f in files)} images). Downloading to {DIR}")
    counts = {"ok": 0, "skip": 0, "missing": 0, "failed": 0}
    session = requests.Session()
    with ThreadPoolExecutor(8) as pool:
        jobs = [pool.submit(fetch, s, n, session) for s, n in files]
        for i, job in enumerate(as_completed(jobs), 1):
            counts[job.result()] += 1
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  new {counts['ok']}  already {counts['skip']}  "
                      f"missing {counts['missing']}  failed {counts['failed']}", flush=True)
    (DIR / "test.jpg").unlink(missing_ok=True)
    for subset in sorted({s for s, _ in files}):
        print(f"  {subset}: {len(list((DIR / subset).glob('*.jpg')))} images")
    if counts["failed"]:
        print("Some files failed: run the same command again to retry them.")


if __name__ == "__main__":
    main()
