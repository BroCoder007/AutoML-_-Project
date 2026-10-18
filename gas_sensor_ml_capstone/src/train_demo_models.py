"""Build the regression, clustering, and sample-reading app artifacts offline."""

from pathlib import Path

from demo_models import save_demo_models


if __name__ == "__main__":
    save_demo_models(Path(__file__).resolve().parents[1])
