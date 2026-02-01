from typing import List, Dict, Any
import datetime

def run_monthly_predictions() -> List[Dict[str, Any]]:
    """
    Mock implementation of run_monthly_predictions for API startup.
    In a real scenario, this would trigger a batch inference job.
    """
    # Mock return data
    return [
        {
            "id": "mock-inference-1",
            "predicted_price": 450000.0,
            "timestamp": datetime.datetime.now().isoformat()
        },
        {
            "id": "mock-inference-2",
            "predicted_price": 520000.0,
            "timestamp": datetime.datetime.now().isoformat()
        }
    ]
