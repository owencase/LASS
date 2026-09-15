#!/usr/bin/env python3
"""Report or remove packages left behind by older LASS dependency sets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Set

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


RUNTIME_ROOTS = {
    "accelerate",
    "faster-whisper",
    "funasr",
    "huggingface-hub",
    "packaging",
    "pip",
    "pyloudnorm",
    "pypdf",
    "pyyaml",
    "setuptools",
    "streamlit",
    "streamlit-pdf",
    "torch",
    "torchaudio",
    "transformers",
    "urllib3",
    "wheel",
}


def installed_distributions() -> Dict[str, metadata.Distribution]:
    return {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }


def dependency_closure(
    distributions: Dict[str, metadata.Distribution], roots: Iterable[str]
) -> Set[str]:
    environment = default_environment()
    required: Set[str] = set()
    pending = [canonicalize_name(name) for name in roots]
    while pending:
        name = canonicalize_name(pending.pop())
        if name in required:
            continue
        required.add(name)
        distribution = distributions.get(name)
        if distribution is None:
            continue
        for specification in distribution.requires or ():
            try:
                requirement = Requirement(specification)
            except ValueError:
                continue
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            pending.append(requirement.name)
    return required


def distribution_size(distribution: metadata.Distribution) -> int:
    total = 0
    for relative_path in distribution.files or ():
        path = Path(distribution.locate_file(relative_path))
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def size_label(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="uninstall the reported orphan distributions from this Python environment",
    )
    args = parser.parse_args()
    distributions = installed_distributions()
    required = dependency_closure(distributions, RUNTIME_ROOTS)
    orphans = sorted(set(distributions) - required)
    if not orphans:
        print("Runtime is already minimal.")
        return 0
    reclaimable = sum(distribution_size(distributions[name]) for name in orphans)
    print(f"Legacy packages: {len(orphans)} ({size_label(reclaimable)} reclaimable)")
    for name in orphans:
        print(f"- {name}: {size_label(distribution_size(distributions[name]))}")
    if not args.apply:
        print("\nReview the list, then rerun with --apply to remove it.")
        return 0
    command = [sys.executable, "-m", "pip", "uninstall", "-y", *orphans]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
