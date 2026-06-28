import uuid
from datetime import datetime
from typing import List, Dict


def generate_alert(probability: float, raw_record: dict = None, threshold_high=0.85,
                    threshold_med=0.6) -> Dict:
    if probability >= threshold_high:
        severity = "high"
    elif probability >= threshold_med:
        severity = "medium"
    else:
        severity = "low"

    alert = {
        "alert_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "probability": round(float(probability), 4),
        "severity": severity,
        "raw_record": raw_record or {},
    }
    return alert


def generate_alerts_batch(probabilities, raw_records=None) -> List[Dict]:
    raw_records = raw_records or [None] * len(probabilities)
    return [generate_alert(p, r) for p, r in zip(probabilities, raw_records)]


def filter_by_severity(alerts: List[Dict], severity: str) -> List[Dict]:
    return [a for a in alerts if a["severity"] == severity]


if __name__ == "__main__":
    sample_probs = [0.1, 0.65, 0.92, 0.3]
    alerts = generate_alerts_batch(sample_probs)
    for a in alerts:
        print(a)
