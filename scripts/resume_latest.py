#!/usr/bin/env python3
"""Resume training from the newest Lightning last.ckpt under the output folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def find_latest_checkpoint(output_dir: Path, checkpoint_name: str) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    candidates = [path for path in output_dir.rglob(checkpoint_name) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No {checkpoint_name!r} found under {output_dir}. "
            "Expected a path like output/.../checkpoints/last.ckpt."
        )

    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find the newest Lightning last checkpoint in the root output folder "
            "and resume trainer.py from it."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder under the repository root that stores Hydra/Lightning outputs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="last.ckpt",
        help="Checkpoint filename to search for.",
    )
    parser.add_argument(
        "-cn",
        "--config-name",
        default="cir_msiglip",
        help="Hydra config name passed to trainer.py.",
    )
    parser.add_argument(
        "--trainer",
        default="trainer.py",
        help="Training entry point, relative to the repository root.",
    )
    parser.add_argument(
        "--use-current-python",
        action="store_true",
        help="Run trainer.py with the current Python instead of `uv run python`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected checkpoint and command without starting training.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args, hydra_overrides = parser.parse_known_args()

    root = repo_root()
    output_dir = resolve_from_root(args.output_dir, root)
    trainer_path = resolve_from_root(args.trainer, root)
    checkpoint_path = find_latest_checkpoint(output_dir, args.checkpoint_name).resolve()

    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer entry point does not exist: {trainer_path}")

    launcher = [sys.executable] if args.use_current_python else ["uv", "run", "python"]
    command = [
        *launcher,
        str(trainer_path),
        "-cn",
        args.config_name,
        f"++ckpt_path={checkpoint_path}",
        *hydra_overrides,
    ]

    print(f"Selected checkpoint: {checkpoint_path}")
    print("Command:", " ".join(command))

    if args.dry_run:
        return 0

    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
