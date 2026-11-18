import re

# pl. "Please retry in 17.3s" a 429-es üzenetben
RETRY_HINT_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def retry_after_hint(error: Exception) -> float | None:
    match = RETRY_HINT_RE.search(str(error))
    try:
        return float(match.group(1)) if match else None
    except ValueError:
        return None
