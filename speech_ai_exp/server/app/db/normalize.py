from __future__ import annotations

import re


def normalize_customer_id(raw: str) -> str:
    """Normalize spoken or formatted IDs: C-1-9-3-8-2 → C-19382, C001257 → C-001257."""
    text = raw.strip().upper().replace(" ", "")
    if not text:
        return ""
    if text.startswith("C"):
        digits = re.sub(r"\D", "", text[1:])
        if digits:
            if len(digits) == 6:
                return f"C-{digits}"
            trimmed = digits.lstrip("0") or "0"
            return f"C-{trimmed}"
    return raw.strip()


def normalize_email(raw: str) -> str:
    """Normalize spoken email: j dot smith at acme dot com → j.smith@acme.com."""
    text = raw.strip().lower()
    text = text.replace(" at ", "@").replace(" dot ", ".")
    text = re.sub(r"\s+", "", text)
    return text


def order_id_candidates(raw: str) -> list[str]:
    """Build candidate order IDs from spoken or formatted numbers."""
    text = raw.strip()
    if not text:
        return []
    candidates = [text]
    digits = re.sub(r"\D", "", text)
    if digits:
        candidates.append(digits)
        candidates.append(f"ORD-{digits}")
        if not text.upper().startswith("ORD"):
            candidates.append(f"ORD-{digits[:4]}-{digits[4:]}" if len(digits) > 4 else f"ORD-{digits}")
    seen: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen
