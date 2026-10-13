"""Create a uniform random sample of the cleaned fire dataset.

The full ``cleaned_master_fires.csv`` holds tens of millions of rows. When the
API runs without PostgreSQL (``DATA_BACKEND=csv``) it serves data from a much
smaller sample file so it starts in seconds. Run::

    python -m api.tools_make_sample --rows 300000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.settings import load_settings  # noqa: E402

LOGGER = logging.getLogger("geodata.api.sample")
CHUNK_SIZE = 500_000


def make_sample(source: Path, destination: Path, rows: int, seed: int = 42) -> int:
    """Reservoir-style sampling: keep a uniform random subset while streaming."""
    rng = np.random.default_rng(seed)
    kept = pd.DataFrame()
    seen = 0
    for chunk in pd.read_csv(source, chunksize=CHUNK_SIZE, low_memory=False):
        seen += len(chunk)
        chunk = chunk.assign(_priority=rng.random(len(chunk)))
        kept = pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > rows:
            kept = kept.nsmallest(rows, "_priority")
        LOGGER.info("Scanned %s rows, holding %s.", seen, len(kept))
    if kept.empty:
        raise ValueError(f"No rows found in {source}")
    kept = kept.drop(columns="_priority").sort_values(["acq_date", "acq_time"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(destination, index=False)
    meta = {"source": str(source), "source_rows": int(seen), "sample_rows": int(len(kept)), "seed": seed}
    Path(str(destination).replace(".csv", ".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s sampled rows (of %s) to %s", len(kept), seen, destination)
    return len(kept)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=settings.fires_csv)
    parser.add_argument("--destination", type=Path, default=settings.fires_sample_csv)
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Source file not found: {args.source}")
    make_sample(args.source, args.destination, args.rows, args.seed)


if __name__ == "__main__":
    main()
