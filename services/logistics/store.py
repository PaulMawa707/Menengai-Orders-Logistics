"""Trips kept in Supabase (PostgREST over HTTPS, server-side secret key only)."""

import os
import re
from datetime import datetime, timezone

import requests

TABLE = "menengai_trips"
PAGE = 1000
LIST_COLS = (
    "id,resource_id,source_key,source,route_uid,route_name,unit_id,unit_name,warehouse,"
    "day,window_from,window_to,plan,files,created_at,created_by"
)
UUID_RE = re.compile(r"^[0-9a-f-]{36}$", re.IGNORECASE)


class StoreError(RuntimeError):
    pass


class NotFound(StoreError):
    pass


def _settings():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
    return (url, key) if url and key else (None, None)


def configured() -> bool:
    return _settings()[0] is not None


def _headers(extra=None):
    _, key = _settings()
    h = {"apikey": key, "Content-Type": "application/json"}
    # Legacy service-role keys are JWTs and go in Authorization as well; new sb_secret_ keys only in apikey.
    if key.startswith("eyJ"):
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


def _request(method, params=None, json=None, headers=None, what="Saving trips"):
    url, _ = _settings()
    if not url:
        raise StoreError("The app database is not configured (SUPABASE_URL / SUPABASE_SECRET_KEY).")
    res = requests.request(
        method, f"{url}/rest/v1/{TABLE}", params=params, json=json, headers=_headers(headers), timeout=30
    )
    if res.status_code >= 300:
        try:
            msg = res.json().get("message") or res.text
        except ValueError:
            msg = res.text
        raise StoreError(f"{what}: {msg}")
    return res


def _iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()


def unix(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _all_rows(params, what):
    out = []
    start = 0
    while True:
        res = _request(
            "GET", params=params, headers={"Range-Unit": "items", "Range": f"{start}-{start + PAGE - 1}"}, what=what
        )
        rows = res.json()
        out.extend(rows)
        if len(rows) < PAGE:
            return out
        start += PAGE


def list_trips(resource_id: int, day_from: str, day_to: str, with_results=False):
    params = [
        ("select", LIST_COLS + (",result" if with_results else "")),
        ("resource_id", f"eq.{int(resource_id)}"),
        ("day", f"gte.{day_from}"),
        ("day", f"lte.{day_to}"),
        ("removed_at", "is.null"),
        ("order", "window_from.asc,id.asc"),
    ]
    return _all_rows(params, "Loading trips")


def get_trip(resource_id: int, trip_id: str):
    if not UUID_RE.match(trip_id or ""):
        raise NotFound("Trip not found")
    params = [
        ("select", LIST_COLS + ",result"),
        ("id", f"eq.{trip_id}"),
        ("resource_id", f"eq.{int(resource_id)}"),
        ("removed_at", "is.null"),
    ]
    rows = _request("GET", params=params, what="Loading trip").json()
    if not rows:
        raise NotFound("Trip not found")
    return rows[0]


def trip_row(*, resource_id, source, route_uid, route_name, unit_id, unit_name, warehouse, day, start, end, plan, files="", created_by=""):
    return {
        "resource_id": int(resource_id),
        "source_key": f"wialon:{int(resource_id)}:{int(route_uid)}",
        "source": source,
        "route_uid": int(route_uid),
        "route_name": route_name,
        "unit_id": int(unit_id),
        "unit_name": unit_name,
        "warehouse": warehouse,
        "day": day,
        "window_from": _iso(start),
        "window_to": _iso(end),
        "plan": plan,
        "files": files,
        "created_by": created_by,
    }


def insert_trip(row) -> str:
    res = _request(
        "POST",
        params={"on_conflict": "source_key"},
        json=row,
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        what="Saving trip",
    )
    return res.json()[0]["id"]


def insert_missing(rows):
    """Saves trips read from Wialon; one already kept for the same route is left as it is."""
    for i in range(0, len(rows), 100):
        _request(
            "POST",
            params={"on_conflict": "source_key"},
            json=rows[i : i + 100],
            headers={"Prefer": "resolution=ignore-duplicates,return=minimal"},
            what="Saving trips from Wialon",
        )


def save_result(trip_id: str, result: dict):
    try:
        _request(
            "PATCH",
            params={"id": f"eq.{trip_id}"},
            json={"result": result, "result_at": datetime.now(timezone.utc).isoformat()},
            headers={"Prefer": "return=minimal"},
            what="Keeping trip result",
        )
    except Exception as exc:
        print(f"[logistics] could not keep the trip result: {exc}")


def remove_trip(resource_id: int, trip_id: str):
    get_trip(resource_id, trip_id)
    _request(
        "PATCH",
        params={"id": f"eq.{trip_id}"},
        json={"removed_at": datetime.now(timezone.utc).isoformat()},
        headers={"Prefer": "return=minimal"},
        what="Removing trip",
    )
