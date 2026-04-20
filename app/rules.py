from typing import Dict

from app.config import CONFIDENCE_THRESHOLD

INTENT_TO_TEAM = {
    "access_issue": "identity_support",
    "billing_issue": "finance_support",
    "network_issue": "network_operations",
    "bug_report": "application_support",
    "feature_request": "product_team",
    "general_query": "service_desk",
}


def derive_priority(intent: str, sentiment: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return "manual_review"

    if intent in {"access_issue", "billing_issue", "network_issue", "bug_report"} and sentiment == "negative":
        return "high"

    if sentiment == "negative":
        return "medium"

    return "low"


def routing_metadata(intent: str, sentiment: str, confidence: float) -> Dict[str, str]:
    return {
        "priority": derive_priority(intent, sentiment, confidence),
        "routing_team": INTENT_TO_TEAM.get(intent, "service_desk"),
    }
