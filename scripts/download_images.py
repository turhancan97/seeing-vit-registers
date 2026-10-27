"""Download the fixed COCO subset and write images/SOURCES.md."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.data import download_images


def main() -> None:
    """Use exactly the same manifest and configuration as the full pipeline."""
    sources = download_images(load_config())
    print(f"Verified {len(sources)} images; attribution is in images/SOURCES.md")


if __name__ == "__main__":
    main()
