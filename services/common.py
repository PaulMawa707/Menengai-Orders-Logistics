import os
import time
from datetime import datetime

import pytz

WAREHOUSES = {
    "MORL": {"lat": -0.28802969095623043, "lon": 36.04494759379902},
    "AQUAMIST": {"lat": -1.2060092437779735, "lon": 36.657015788043836},
    "STO": {"lat": -1.3029466, "lon": 36.8660916},
}

EAT = pytz.timezone("Africa/Nairobi")

os.environ["TZ"] = "Africa/Nairobi"
try:
    time.tzset()
except Exception:
    pass


def parse_datetime_local(value: str, label: str = "Date/time") -> datetime:
    if not value or not str(value).strip():
        raise ValueError(f"{label} is required.")
    raw = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(raw, fmt)
            return EAT.localize(naive)
        except ValueError:
            continue
    raise ValueError(f"Invalid {label.lower()} format.")


def compute_route_timestamps_from_range(
    from_value: str, to_value: str, from_label="Route From", to_label="Route End"
):
    start_time = parse_datetime_local(from_value, from_label)
    end_time = parse_datetime_local(to_value, to_label)
    tf, tt = int(start_time.timestamp()), int(end_time.timestamp())
    if tt <= tf:
        raise ValueError(f"{to_label} must be after {from_label}.")
    return tf, tt
