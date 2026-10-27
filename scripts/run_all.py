"""Reproduce all results with fresh inference unless --reuse is explicit."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_all


def main() -> None:
    """Expose paths/device overrides without duplicating scientific settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse saved feature arrays for plotting"
    )
    args = parser.parse_args()
    run_all(args.config, args.output_dir, args.image_dir, args.device, args.reuse)


if __name__ == "__main__":
    main()
