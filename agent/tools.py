import json
import re
import os
from typing import Dict, Any, Optional

ORDERS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "orders.json")

EXPLICIT_ORD_RE = re.compile(r"\bORD-(\d{4,6})\b", re.IGNORECASE)
ORD_NO_HYPHEN_RE = re.compile(r"\bORD(\d{4,6})\b", re.IGNORECASE)
ORDER_LABELED_RE = re.compile(
    r"\b(?:ord(?:er)?|package|pkg)\s*(?:id|number|no\.?|#)?\s*(?:is|:)?\s*[#:]?\s*(ORD-?)?(\d{4,6})\b",
    re.IGNORECASE,
)
HASH_ID_RE = re.compile(r"(?<![A-Za-z0-9])#(\d{4,6})\b")
BARE_ID_RE = re.compile(r"^\s*(?:ORD-?)?(\d{4,6})\s*$", re.IGNORECASE)


def load_orders_db() -> Dict[str, Any]:
    """Loads the mock order database from data/orders.json."""
    if not os.path.exists(ORDERS_DB_PATH):
        return {}
    with open(ORDERS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_order_id(raw_digits: str) -> str:
    return f"ORD-{raw_digits}"


def _is_dollar_amount(text: str, digit_start: int) -> bool:
    return digit_start > 0 and text[digit_start - 1] == "$"


def extract_order_id(query: str, *, allow_bare_number: bool = False) -> Optional[str]:
    """
    Extracts an order ID only when the query clearly refers to one.

    Matches: ORD-1001, ORD1001, order #1001, order 1001, #1001.
    Does not treat prices or the word "order" in a policy question as an ID.
    Bare numeric IDs (e.g. "1002") are accepted only during pending clarification.
    """
    if not query:
        return None

    match = EXPLICIT_ORD_RE.search(query)
    if match:
        return _normalize_order_id(match.group(1))

    match = ORD_NO_HYPHEN_RE.search(query)
    if match:
        return _normalize_order_id(match.group(1))

    match = ORDER_LABELED_RE.search(query)
    if match and not _is_dollar_amount(query, match.start(2)):
        return _normalize_order_id(match.group(2))

    match = HASH_ID_RE.search(query)
    if match and not _is_dollar_amount(query, match.start(1)):
        return _normalize_order_id(match.group(1))

    if allow_bare_number:
        match = BARE_ID_RE.fullmatch(query.strip())
        if match:
            return _normalize_order_id(match.group(1))

    return None


def check_order_status(order_id: str) -> Dict[str, Any]:
    """Look up a mock order. Never invents records for unknown IDs."""
    if not order_id or not isinstance(order_id, str):
        return {
            "found": False,
            "error": "No order ID provided. Please provide a valid order ID.",
        }

    extracted = extract_order_id(order_id, allow_bare_number=True)
    clean_id = extracted or order_id.strip().upper()
    if clean_id.isdigit():
        clean_id = _normalize_order_id(clean_id)

    orders_db = load_orders_db()
    if clean_id in orders_db:
        order = orders_db[clean_id]
        return {
            "found": True,
            "order_id": order["order_id"],
            "customer_name": order.get("customer_name", "Valued Customer"),
            "status": order["status"],
            "eta": order["eta"],
            "items": order["items"],
            "shipping_address": order.get("shipping_address", "N/A"),
            "tracking_number": order.get("tracking_number", "N/A"),
        }

    return {
        "found": False,
        "order_id": clean_id,
        "error": f"Order '{clean_id}' was not found in our system. Please check your order number.",
    }
