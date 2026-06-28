import re
from typing import List, Dict

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[threat_nlp_analyzer] تحذير: pip install transformers")

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DOMAIN_PATTERN = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
MD5_PATTERN = re.compile(r"\b[a-fA-F0-9]{32}\b")
SHA256_PATTERN = re.compile(r"\b[a-fA-F0-9]{64}\b")


def extract_iocs_regex(text: str) -> Dict[str, List[str]]:
    return {
        "ips": list(set(IP_PATTERN.findall(text))),
        "domains": list(set(DOMAIN_PATTERN.findall(text))),
        "md5_hashes": list(set(MD5_PATTERN.findall(text))),
        "sha256_hashes": list(set(SHA256_PATTERN.findall(text))),
    }


class ThreatNLPAnalyzer:
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            self.ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english",
                                 aggregation_strategy="simple")
            self.summarizer = pipeline("summarization")
        else:
            self.ner = None
            self.summarizer = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        if not self.ner:
            return {"organizations": [], "locations": []}
        results = self.ner(text)
        orgs = list({r["word"] for r in results if r["entity_group"] == "ORG"})
        locs = list({r["word"] for r in results if r["entity_group"] == "LOC"})
        return {"organizations": orgs, "locations": locs}

    def summarize(self, text: str, max_length: int = 80) -> str:
        if not self.summarizer:
            return text[:200] + "..."
        result = self.summarizer(text[:2000], max_length=max_length, min_length=20,
                                  do_sample=False)
        return result[0]["summary_text"]

    def analyze_report(self, text: str) -> Dict:
        return {
            "entities": self.extract_entities(text),
            "iocs_regex": extract_iocs_regex(text),
            "summary": self.summarize(text),
        }


def compare_nlp_vs_regex(reports: List[str], analyzer: "ThreatNLPAnalyzer") -> Dict:
    total_iocs_regex = 0
    total_entities_nlp = 0
    for r in reports:
        iocs = extract_iocs_regex(r)
        total_iocs_regex += sum(len(v) for v in iocs.values())
        entities = analyzer.extract_entities(r)
        total_entities_nlp += sum(len(v) for v in entities.values())
    return {
        "total_iocs_found_by_regex": total_iocs_regex,
        "total_entities_found_by_nlp": total_entities_nlp,
        "conclusion": (
            "Regex أسرع وأدق لاستخراج IOCs (IP/hash/domain) لأنها أنماط ثابتة الشكل. "
            "NLP/NER أفضل لفهم السياق: من هي الجهة المهاجمة، أي دولة مستهدفة، "
            "وأي تنظيم مذكور — معلومات لا يستطيع regex استخراجها."
        ),
    }


if __name__ == "__main__":
    sample_report = (
        "APT29, linked to Russia, targeted servers at 192.168.1.10 and "
        "malicious-domain.com. The malware hash was "
        "44d88612fea8a8f36de82e1278abb02f. Researchers at Microsoft observed "
        "the campaign affecting organizations in the United States and Germany."
    )
    analyzer = ThreatNLPAnalyzer()
    result = analyzer.analyze_report(sample_report)
    print(result)
