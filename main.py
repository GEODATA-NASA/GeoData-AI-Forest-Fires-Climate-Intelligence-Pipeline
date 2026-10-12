import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CLEANED_DATA_DIR
from integration.NASA_POWER import download_weather_data
from loadation.load import load_data_to_postgres
from modeling.train_model import main as train_model
from transformation.transform import transform_fire_data, transform_weather_data

LOGGER = logging.getLogger("geodata.pipeline")


def run_pipeline():
    """Run integration, transformation, loading, then model training."""
    LOGGER.info("Starting the GEODATA pipeline.")

    download_weather_data()

    transform_fire_data(collect_data=False)
    transform_weather_data()

    fire_file = Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv"
    weather_file = Path(CLEANED_DATA_DIR) / "cleaned_weather.csv"
    missing_files = [str(path) for path in (fire_file, weather_file) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Transformation did not produce required cleaned files: "
            + ", ".join(missing_files)
        )

    load_data_to_postgres()

    training_status = train_model()
    if training_status != 0:
        raise RuntimeError(f"Modeling stage failed with status {training_status}.")

    LOGGER.info("The GEODATA pipeline completed successfully.")


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_pipeline()
    except Exception:
        LOGGER.exception("The GEODATA pipeline failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
