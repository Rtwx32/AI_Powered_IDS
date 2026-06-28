import random
from datetime import datetime
from typing import List, Dict


PHISHING_SCENARIO_TEMPLATES = [
    "Urgency-based: ادعاء أن الحساب سيُقفل خلال 24 ساعة إن لم يضغط المستخدم على رابط.",
    "Authority-based: انتحال شخصية IT Support يطلب 'تحديث' بيانات الدخول.",
    "Reward-based: إشعار وهمي بفوز أو استرداد مالي يتطلب 'تأكيد' المعلومات.",
]


def social_engineering_scenarios() -> List[str]:
    return PHISHING_SCENARIO_TEMPLATES


def vulnerability_scan_logic_overview() -> str:
    return (
        "1. Asset discovery: اكتشاف الأجهزة والخدمات النشطة على الشبكة\n"
        "2. Service fingerprinting: تحديد نوع/إصدار كل خدمة\n"
        "3. Matching against CVE database: مطابقة الإصدارات بثغرات معروفة\n"
        "4. Risk scoring: ترتيب الثغرات حسب CVSS score وقابلية الاستغلال"
    )


def alert_priority_score(alert: Dict) -> float:
    score = 0.0
    score += alert.get("model_probability", 0) * 0.5
    score += (1 if alert.get("asset_critical") else 0) * 0.3
    score += min(alert.get("failed_logins", 0) / 10, 1) * 0.2
    return round(min(score, 1.0), 3)


def triage_alerts(alerts: List[Dict]) -> List[Dict]:
    for a in alerts:
        a["priority_score"] = alert_priority_score(a)
    return sorted(alerts, key=lambda x: x["priority_score"], reverse=True)


def user_behavior_baseline(login_history: List[Dict]) -> Dict:
    hours = [h["hour"] for h in login_history]
    countries = [h["country"] for h in login_history]
    return {
        "typical_hours": sorted(set(hours)),
        "typical_countries": list(set(countries)),
    }


def detect_uba_anomaly(login_event: Dict, baseline: Dict) -> bool:
    hour_anomaly = login_event["hour"] not in baseline["typical_hours"]
    country_anomaly = login_event["country"] not in baseline["typical_countries"]
    return hour_anomaly or country_anomaly


def automated_playbook_trigger(alert: Dict) -> str:
    if alert.get("priority_score", 0) >= 0.8:
        return f"🚨 ALERT {alert.get('alert_id', '?')}: triggering automated IR playbook now."
    return f"ℹ️ ALERT {alert.get('alert_id', '?')}: logged for manual review."


if __name__ == "__main__":
    print("=== Red Team ===")
    for s in social_engineering_scenarios():
        print("-", s)

    print("\n=== Blue Team: Alert Triage ===")
    demo_alerts = [
        {"alert_id": "A1", "model_probability": 0.9, "asset_critical": True, "failed_logins": 5},
        {"alert_id": "A2", "model_probability": 0.4, "asset_critical": False, "failed_logins": 1},
        {"alert_id": "A3", "model_probability": 0.7, "asset_critical": True, "failed_logins": 8},
    ]
    triaged = triage_alerts(demo_alerts)
    for a in triaged:
        print(a, "→", automated_playbook_trigger(a))

    print("\n=== Blue Team: UBA ===")
    history = [{"hour": 9, "country": "SA"}, {"hour": 10, "country": "SA"}]
    baseline = user_behavior_baseline(history)
    new_login = {"hour": 3, "country": "RU"}
    print("Baseline:", baseline)
    print("Anomaly detected:", detect_uba_anomaly(new_login, baseline))
