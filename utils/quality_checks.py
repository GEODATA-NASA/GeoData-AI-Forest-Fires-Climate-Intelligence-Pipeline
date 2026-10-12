import pandas as pd
import logging

logger = logging.getLogger("GeoDataAI.QualityCheck")

def validate_fire_data(
    df: pd.DataFrame, *, raise_if_empty: bool = True
) -> pd.DataFrame:
    """Perform data quality checks on fire data"""
    initial_count = len(df)
    logger.info(f"Starting data quality checks on {initial_count} rows...")

    # 1. Make sure latitude and longitude are within valid ranges
    valid_coords = (
        df["latitude"].between(-90, 90) & 
        df["longitude"].between(-180, 180)
    )
    df = df[valid_coords]

    # 2. Ensure brightness and frp values are non-negative and logical
    if "brightness" in df.columns:
        df = df[df["brightness"] > 0]
    if "frp" in df.columns:
        df = df[df["frp"] >= 0]

    # 3. Remove completely duplicated rows
    df = df.drop_duplicates()

    final_count = len(df)
    dropped_rows = initial_count - final_count
    
    logger.warning(f"Data Quality Check completed. Dropped {dropped_rows} invalid/corrupted rows. Remaining: {final_count} rows.")
    
    if final_count == 0 and raise_if_empty:
        raise ValueError("Critical Error: All rows failed data quality checks!")
        
    return df