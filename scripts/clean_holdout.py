"""
Generate cleaning_holdout.csv by applying the same cleaning steps from 01_EDA_cleaning.ipynb
"""
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Load datasets
print("Loading holdout dataset...")
holdout_df = pd.read_csv(RAW_DIR / "holdout.csv")
metros = pd.read_csv(RAW_DIR / "usmetros.csv")

print(f"Original holdout shape: {holdout_df.shape}")

# City name mapping (same as in 01_EDA_cleaning.ipynb)
city_mapping = {
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont',
    'DC_Metro': 'Washington-Arlington-Alexandria',
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell'
}


def clean_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Apply city name fixes, merge lat/lng from metros, drop dup col."""
    df["city_full"] = df["city_full"].replace(city_mapping)

    # Create a join key in metros by removing the state suffix (e.g. ", GA")
    metros_clean = metros.copy()
    metros_clean["metro_join"] = metros_clean["metro_full"].astype(str).str.split(',').str[0]

    df = df.merge(
        metros_clean[["metro_join", "lat", "lng"]],
        how="left",
        left_on="city_full",
        right_on="metro_join"
    )
    df.drop(columns=["metro_join"], inplace=True)

    # Log any cities that still didn't match
    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("[WARNING] Still missing lat/lng for:", missing)
    else:
        print("[INFO] All cities matched with metros dataset.")

    return df


# Apply cleaning + merge
print("\nApplying city name standardization and geolocation merge...")
holdout_df = clean_and_merge(holdout_df)
print(f"After merge shape: {holdout_df.shape}")

# Remove duplicates (excluding date/year columns)
print("\nRemoving duplicates...")
duplicated_before = holdout_df[holdout_df.duplicated(subset=holdout_df.columns.difference(['date', 'year']))].shape[0]
print(f"Duplicated rows (excluding date column): {duplicated_before}")

holdout_df = holdout_df.drop_duplicates(subset=holdout_df.columns.difference(['date', 'year']), keep=False)
print(f"After duplicate removal shape: {holdout_df.shape}")

# Clean outliers above 19M
print("\nCleaning outliers (median_list_price > 19M)...")
outliers_count = (holdout_df['median_list_price'] > 19_000_000).sum()
print(f"Outliers found: {outliers_count}")

holdout_df = holdout_df[holdout_df['median_list_price'] <= 19_000_000].copy()
print(f"Final shape: {holdout_df.shape}")

# Save cleaned dataset
output_path = PROCESSED_DIR / "cleaning_holdout.csv"
holdout_df.to_csv(output_path, index=False)
print(f"\n[INFO] Saved cleaned holdout dataset to: {output_path}")
