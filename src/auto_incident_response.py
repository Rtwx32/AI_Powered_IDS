import random
from datetime import datetime
from typing import Dict, List


def extract_ip(alert: Dict) -> str:
    return alert.get("raw_record", {}).get("src_ip", "0.0.0.0")


def osint_analyzer(ip: str) -> Dict:
    random.seed(hash(ip) % 1000)
    return {
        "ip": ip,
        "is_known_malicious": random.random() > 0.7,
        "reports_count": random.randint(0, 50),
        "country": random.choice(["RU", "CN", "SA", "US", "Unknown"]),
    }


def virustotal_check(ip: str) -> Dict:
    random.seed(hash(ip) % 500)
    malicious_votes = random.randint(0, 10)
    return {"ip": ip, "malicious_engines": malicious_votes, "total_engines": 90}


def calculate_risk(osint_info: Dict, vt_info: Dict) -> float:
    score = 0.0
    score += 0.4 if osint_info["is_known_malicious"] else 0
    score += min(osint_info["reports_count"] / 50, 1) * 0.2
    score += min(vt_info["malicious_engines"] / vt_info["total_engines"], 1) * 0.4
    return round(min(score, 1.0), 3)


def block_ip(ip: str) -> str:
    return f"[ACTION] تم حظر {ip} على الفايروول (محاكاة)."


def create_ticket(alert: Dict, osint_info: Dict, score: float) -> Dict:
    return {
        "ticket_id": f"TCK-{random.randint(1000, 9999)}",
        "created_at": datetime.now().isoformat(),
        "alert_id": alert.get("alert_id"),
        "risk_score": score,
        "summary": f"تنبيه أمني من {osint_info['ip']} (دولة: {osint_info['country']})",
        "status": "open",
    }


def handle_incident(alert: Dict, auto_block_threshold: float = 0.8) -> Dict:
    ip = extract_ip(alert)
    osint_info = osint_analyzer(ip)
    vt_info = virustotal_check(ip)
    score = calculate_risk(osint_info, vt_info)

    actions_taken = []
    if score >= auto_block_threshold:
        actions_taken.append(block_ip(ip))

    ticket = create_ticket(alert, osint_info, score)

    return {
        "ip": ip,
        "risk_score": score,
        "osint": osint_info,
        "virustotal": vt_info,
        "actions_taken": actions_taken,
        "ticket": ticket,
    }


def simulate_incidents(n: int = 10) -> List[Dict]:
    results = []
    for i in range(n):
        fake_alert = {
            "alert_id": f"AL-{i}",
            "raw_record": {"src_ip": f"203.0.113.{random.randint(1, 254)}"},
        }
        results.append(handle_incident(fake_alert))
    return results


if __name__ == "__main__":
    results = simulate_incidents(10)
    for r in results:
        blocked = "🚫 BLOCKED" if r["actions_taken"] else "📋 ticket only"
        print(f"{r['ip']} | risk={r['risk_score']} | {blocked} | {r['ticket']['ticket_id']}")
