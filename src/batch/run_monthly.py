from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import datetime
import uuid

from src.inference_pipeline.inference import predict

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaning_holdout.csv"  # Use holdout as "new data"
PREDICTIONS_DIR = DATA_DIR / "predictions"

def run_monthly_predictions() -> List[Dict[str, Any]]:
    """
    Simulate a monthly batch prediction cycle using available test data.

    Responsibility:
        - Orchestrates the loading of 'new' data from the holdout split.
        - Invokes the inference pipeline for record-level predictions.
        - Persists batch results to the historical predictions audit log.
        - Prepares a formatted summary for API or reporting downstream consumers.
    """
    print(f"🚀 Starting batch run at {datetime.datetime.now()}")

    # 1. Load Data
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Input data not found at {PROCESSED_DATA_PATH}. Please run feature pipeline first.")

    # Simulating 'new' data - let's take a sample or the whole set
    input_df = pd.read_csv(PROCESSED_DATA_PATH)

    # 2. Run Inference
    # Predict function handles preprocessing internally if we pass raw-like data
    # (Note: cleaning_holdout is 'cleaned' but not 'feature engineered' yet, which matches inference input expectation)
    try:
        preds_df = predict(input_df)
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise e

    # 3. Save Output
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PREDICTIONS_DIR / f"preds_{timestamp}.csv"

    preds_df.to_csv(output_file, index=False)
    print(f"✅ Batch predictions saved to {output_file}")

    # 4. Return results (preview)
    # Convert to list of dicts for API response
    # Add a unique ID and timestamp for tracking if not present
    results = preds_df.head(100).to_dict(orient="records")
    for res in results:
        res["batch_run_id"] = f"batch_{timestamp}"
        res["processed_at"] = datetime.datetime.now().isoformat()

    return results

if __name__ == "__main__":
    run_monthly_predictions()
