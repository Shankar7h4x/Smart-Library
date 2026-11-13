"""Pure validation helpers for occupancy observations."""
from datetime import datetime, timedelta


def is_fresh_timestamp(observed_at: datetime, now: datetime, max_age: timedelta = timedelta(minutes=30)) -> bool:
    """Return true only for non-future observations within the freshness window."""
    if observed_at is None or now is None:
        return False
    if observed_at.tzinfo is None and now.tzinfo is not None:
        observed_at = observed_at.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and observed_at.tzinfo is not None:
        now = now.replace(tzinfo=observed_at.tzinfo)
    age = now - observed_at
    return timedelta(0) <= age <= max_age


def valid_occupancy(value) -> bool:
    """Check that an occupancy value is numeric, finite and within 0–100."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf") and 0.0 <= number <= 100.0
