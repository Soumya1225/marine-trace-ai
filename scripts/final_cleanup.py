"""Final tidy-up of the project folder before submission / GitHub upload. Nothing is deleted except caches:
leftovers are MOVED to ../old_unused/final_cleanup/ so they can be restored.

    python scripts/final_cleanup.py        (run from the project folder with the app stopped)
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "old_unused" / "final_cleanup"
LEFTOVERS = ["colab_upload", "backup_300cases", ".pytest_cache", "frontend/dist", "nohup.out"]
CACHE_SKIP = {"venv", "node_modules", ".git"}
GITIGNORE_ADD = ["Datasets/feedback/", "__pycache__/", ".pytest_cache/", "*.log", "frontend/dist/"]


def main():
    moved = []
    for rel in LEFTOVERS:
        p = ROOT / rel
        if p.exists():
            dst = DEST / rel.replace("/", "_")
            if dst.exists():
                dst = DEST / (dst.name + "_2")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved.append(rel)
    print(f"Moved to {DEST}: {', '.join(moved) or 'nothing'}")

    caches = 0
    for p in list(ROOT.rglob("__pycache__")):
        if not any(part in CACHE_SKIP for part in p.relative_to(ROOT).parts):
            shutil.rmtree(p, ignore_errors=True)
            caches += 1
    print(f"Removed {caches} __pycache__ folders (Python recreates them automatically)")

    gi = ROOT / ".gitignore"
    lines = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    added = [x for x in GITIGNORE_ADD if x not in lines]
    if added:
        gi.write_text("\n".join(lines + ["", "# added by final_cleanup"] + added) + "\n", encoding="utf-8")
    print(f".gitignore: added {', '.join(added) or 'nothing'}")

    print("\nProject folder now:")
    for p in sorted(ROOT.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if p.name in ("venv",):
            print(f"  {p.name}/  (local Python environment, not uploaded)")
        elif p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file()) if p.name != "frontend" else "-"
            print(f"  {p.name}/" + (f"  ({n} files)" if n != "-" else ""))
        else:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
