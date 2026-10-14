"""Apply a Git patch only after a successful source/configuration backup."""
import argparse
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.backup import backup_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    patch = args.patch.resolve()
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    print(backup_sources(root, args.backup_dir))
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)


if __name__ == "__main__":
    main()
