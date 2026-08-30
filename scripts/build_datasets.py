"""CLI: build trusted and refined datasets from raw xlsx."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_predictor.config import settings
from churn_predictor.data.pipeline import DataPipeline


def main() -> None:
    pipeline = DataPipeline()
    pipeline.run(
        src_xlsx=settings.raw_data_path,
        trusted_path=settings.trusted_data_path,
        refined_path=settings.refined_data_path,
    )
    print(f"trusted → {settings.trusted_data_path}")
    print(f"refined → {settings.refined_data_path}")


if __name__ == "__main__":
    main()
