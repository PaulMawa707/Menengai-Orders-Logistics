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


def _same_trip(have, row) -> bool:
    plan, new = have.get("plan") or {}, row["plan"]
    return (
        int(have["unit_id"]) == row["unit_id"]
        and have.get("unit_name") == row["unit_name"]
        and have["route_name"] == row["route_name"]
        and unix(have["window_from"]) == unix(row["window_from"])
        and unix(have["window_to"]) == unix(row["window_to"])
        and plan.get("km") == new.get("km")
        and _same_path(have, row)
        and [(s.get("name"), s.get("eta")) for s in plan.get("stops", [])] == [(s["name"], s["eta"]) for s in new["stops"]]
    )


ROAD_ROUTINGS = ("wialon_road", "wialon_osrm")


def _same_path(have, row) -> bool:
    a, b = (have.get("plan") or {}).get("geometry") or [], row["plan"].get("geometry") or []
    return len(a) == len(b) and a[:1] == b[:1] and a[-1:] == b[-1:]


STATUS_KEYS = ("ws", "wat", "wleft")


def _status_of(plan):
    return [tuple(s.get(k) for k in STATUS_KEYS) for s in plan.get("stops", [])], plan.get("wStart"), plan.get("wReturn")


def with_status(old_plan, new_plan):
    """`old_plan` with Wialon's fulfilment (order status, arrival, departure) taken from `new_plan`.

    Stops are matched in order when both have the same number of stops, otherwise by name.
    """
    new_stops = new_plan.get("stops", [])
    old_stops = old_plan.get("stops", [])
    if len(new_stops) == len(old_stops):
        source = new_stops
    else:
        by_name = {s["name"].strip().lower(): s for s in new_stops}
        source = [by_name.get(s["name"].strip().lower(), {}) for s in old_stops]
    stops = [{**{k: v for k, v in o.items() if k not in STATUS_KEYS}, **{k: n[k] for k in STATUS_KEYS if k in n}} for o, n in zip(old_stops, source)]
    return {**old_plan, "stops": stops, "wStart": new_plan.get("wStart"), "wReturn": new_plan.get("wReturn")}


def refresh_unfinished(rows):
    """Brings trips up to date with Wialon Logistics. Routes read from Wialon follow it fully while they have no final
    result (running routes can still be edited there); afterwards only the planned road and Wialon's fulfilment.
    Uploads keep the plan made by this app and only take Wialon's fulfilment."""
    for i in range(0, len(rows), 50):
        part = {r["source_key"]: r for r in rows[i : i + 50]}
        keys = ",".join(f'"{k}"' for k in part)
        have = _request(
            "GET",
            params=[
                ("select", "id,source_key,source,unit_id,unit_name,route_name,window_from,window_to,plan,result_at"),
                ("source_key", f"in.({keys})"),
                ("removed_at", "is.null"),
            ],
            what="Loading trips from Wialon",
        ).json()
        for h in have:
            row = part.get(h["source_key"])
            if not row:
                continue
            old = h.get("plan") or {}
            if h.get("source") == "upload":
                if _status_of(old) == _status_of(with_status(old, row["plan"])):
                    continue
                patch = {"plan": with_status(old, row["plan"])}
            else:
                if row["plan"].get("routing") == "wialon" and old.get("routing") in ROAD_ROUTINGS:
                    # Wialon gave no road path; keep the one already worked out for this trip.
                    row = {**row, "plan": {**row["plan"], "geometry": old.get("geometry"), "routing": old["routing"]}}
                if not h.get("result_at") and not (_same_trip(h, row) and _status_of(old) == _status_of(row["plan"])):
                    patch = {k: row[k] for k in ("unit_id", "unit_name", "route_name", "warehouse", "day", "window_from", "window_to", "plan")}
                elif not _same_path(h, row) or _status_of(old) != _status_of(with_status(old, row["plan"])):
                    patch = {"plan": {**with_status(old, row["plan"]), "geometry": row["plan"]["geometry"], "routing": row["plan"]["routing"]}}
                else:
                    continue
            _request(
                "PATCH",
                params={"id": f"eq.{h['id']}"},
                json=patch,
                headers={"Prefer": "return=minimal"},
                what="Updating a trip from Wialon",
            )


def update_plan(trip_id: str, plan: dict):
    _request(
        "PATCH",
        params={"id": f"eq.{trip_id}"},
        json={"plan": plan},
        headers={"Prefer": "return=minimal"},
        what="Saving the planned road",
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
