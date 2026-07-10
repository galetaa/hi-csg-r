#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from src.pipeline.data_acquisition import (
    AcquisitionResult,
    acquisition_plan,
    download_datasets,
    extract_datasets,
    manual_check,
    write_local_checksums,
)
from src.pipeline.registry import REPO_ROOT


def _print(results: list[AcquisitionResult]) -> int:
    failed = 0
    for result in results:
        status = result.status
        print(f"{status} {result.dataset_id}: {result.message}")
        failed += status == "ERROR"
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire HI-CSG-R datasets from the registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("plan", "download", "extract", "manual-check"):
        item = subparsers.add_parser(command)
        item.add_argument("datasets", nargs="*")
        if command in {"download", "extract"}:
            item.add_argument("--execute", action="store_true")
        if command == "download":
            item.add_argument("--force", action="store_true")
            item.add_argument("--workers", type=int, default=4)

    checksum = subparsers.add_parser("checksum")
    checksum.add_argument("datasets", nargs="+")

    args = parser.parse_args(argv)
    if args.command == "plan":
        return _print(acquisition_plan(args.datasets))
    if args.command == "download":
        return _print(
            download_datasets(
                args.datasets,
                execute=args.execute,
                force=args.force,
                workers=args.workers,
            )
        )
    if args.command == "extract":
        return _print(extract_datasets(args.datasets, execute=args.execute))
    if args.command == "manual-check":
        return _print(manual_check(args.datasets))
    if args.command == "checksum":
        for dataset_id in args.datasets:
            print(write_local_checksums(dataset_id).relative_to(REPO_ROOT))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
