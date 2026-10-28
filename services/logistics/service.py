import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from services import assets

from . import store, wialon
from .plan import RULES, build_plan
from .track import LEAD_S, TAIL_S, TRIP_STATES, progress_of, track_plan, trip_state

TZ_S = 3 * 3600
EAT = timezone(timedelta(seconds=TZ_S))
# Late position uploads (vehicles that were out of coverage) can still change a trip for a while after it ends.
SETTLE_S = 3600
# Wialon history is searched by when routes finished, which can be well after the day they ran.
BEFORE_S = 2 * 86400
AFTER_S = 30 * 86400
# A period already read from Wialon history on this server is not read again for this long.
FRESH_S = 300
# Wialon order flags: 0x100 marks a warehouse (start 260, end 264).
WAREHOUSE_FLAG = 0x100
WORKERS = 4
# Wialon refuses more than about 10 concurrent requests per user.
REPORT_WORKERS = 6

_synced = {}
_sync_lock = threading.Lock()


def local_day(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=EAT).strftime("%Y-%m-%d")


def day_start(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=EAT).timestamp())


def _num(v):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return 0.0
    return n if n == n else 0.0


# ---- Uploads ----


def record_upload(*, resource_id, route_uid, route_name, unit_id, unit_name, warehouse, depot, stops, start, end, files, created_by):
    """Keeps a dispatched upload as a trip; its stops are in the order sent to Wialon."""
    plan = build_plan(stops, {"name": warehouse, **depot}, start, end)
    row = store.trip_row(
        resource_id=resource_id,
        source="upload",
        route_uid=route_uid,
        route_name=route_name,
        unit_id=unit_id,
        unit_name=unit_name,
        warehouse=warehouse,
        day=local_day(start),
        start=start,
        end=end,
        plan=plan,
        files=files,
        created_by=created_by,
    )
    return store.insert_trip(row)


# ---- Finished routes from Wialon Logistics ----


def _wialon_plan(orders):
    """A route dispatched in Wialon Logistics as a trip: its orders become the stops, with Wialon's planned times."""
    pts = [o for o in orders if (o.get("y") or o.get("x")) and o.get("y") is not None and o.get("x") is not None]
    pts.sort(key=lambda o: ((o.get("p") or {}).get("r") or {}).get("i", 0))
    if not pts:
        return None
    depot = None
    stops = []
    km = drive = service = 0.0
    return_at = 0
    for o in pts:
        p = o.get("p") or {}
        r = p.get("r") or {}
        eta = int(r.get("vt") or o.get("tf") or 0)
        km += _num(r.get("m"))
        drive += _num(r.get("t"))
        if int(o.get("f") or 0) & WAREHOUSE_FLAG:
            if depot is None:
                depot = {"name": o.get("n") or "Warehouse", "lat": float(o["y"]), "lon": float(o["x"])}
            else:
                return_at = max(return_at, eta)
            continue
        svc = _num(p.get("ut"))
        service += svc
        stops.append(
            {
                "seq": len(stops) + 1,
                "name": (o.get("n") or "").strip() or f"Order {len(stops) + 1}",
                "location": p.get("a") or "",
                "lat": float(o["y"]),
                "lon": float(o["x"]),
                "amount": round(_num(p.get("cost") or p.get("c")), 2),
                "weightKg": round(_num(p.get("weight") or p.get("w")), 1),
                "invoice": "",
                "legKm": round(_num(r.get("m")) / 10) / 100,
                "eta": eta,
                "etd": eta + int(svc),
            }
        )
    if not stops:
        return None
    # Routes copied in Wialon keep the original day's order windows (tf/tt); planned visit times are current.
    visits = [int(((o.get("p") or {}).get("r") or {}).get("vt") or 0) for o in pts]
    visits = [v for v in visits if v]
    if not visits:
        return None
    start = min(visits)
    return_at = return_at or stops[-1]["etd"]
    deadlines = [int(o.get("tt") or 0) for o in pts if start < int(o.get("tt") or 0) <= start + 2 * 86400]
    end = max(deadlines) if deadlines else return_at
    geometry = ([[depot["lat"], depot["lon"]]] if depot else []) + [[s["lat"], s["lon"]] for s in stops] + ([[depot["lat"], depot["lon"]]] if depot else [])
    return {
        "depot": depot,
        "start": start,
        "end": max(end, start),
        "returnAt": return_at,
        "km": round(km / 100) / 10,
        "driveMin": round(drive / 60),
        "serviceMin": round(service / 60),
        "stops": stops,
        "routing": "wialon",
        "geometry": geometry,
        "overrun": return_at > end,
        "rules": dict(RULES),
    }


def wialon_trip_rows(resource_id, routes, orders, day_from, day_to, names):
    by_uid = {o["uid"]: o for o in orders if o.get("uid")}
    groups = {}
    for o in by_uid.values():
        rid = ((o.get("p") or {}).get("r") or {}).get("id")
        if rid:
            groups.setdefault(rid, {})[o["uid"]] = o
    route_by_uid = {r["uid"]: r for r in routes if r.get("uid")}
    for r in route_by_uid.values():
        for u in r.get("ord") or []:
            if u in by_uid:
                groups.setdefault(r["uid"], {})[u] = by_uid[u]
    rows = []
    for rid, group in groups.items():
        route = route_by_uid.get(rid) or {}
        lst = list(group.values())
        units = Counter(int(o.get("u") or (o.get("p") or {}).get("u") or 0) for o in lst)
        units.pop(0, None)
        unit_id = int((route.get("st") or {}).get("u") or 0) or (units.most_common(1)[0][0] if units else 0)
        if not unit_id:
            continue
        plan = _wialon_plan(lst)
        if not plan:
            continue
        day = local_day(plan["start"])
        if day < day_from or day > day_to:
            continue
        unit_name = names.get(unit_id) or f"Unit {unit_id}"
        rows.append(
            store.trip_row(
                resource_id=resource_id,
                source="wialon",
                route_uid=rid,
                route_name=(route.get("n") or "").strip() or f"{unit_name} - Wialon route {day}",
                unit_id=unit_id,
                unit_name=unit_name,
                warehouse=(plan["depot"] or {}).get("name", ""),
                day=day,
                start=plan["start"],
                end=plan["end"],
                plan=plan,
                created_by="Wialon Logistics",
            )
        )
    return rows


def _unit_names(token, unit_ids):
    names = {}
    missing = []
    for u in unit_ids:
        a = assets.get_asset_by_item_id(u)
        if a:
            names[u] = a["name"]
        else:
            missing.append(u)
    if missing:
        try:
            names.update(wialon.unit_names(token, missing))
        except Exception as exc:
            print(f"[logistics] could not read unit names: {exc}")
    return names


def sync_wialon_history(token, resource_id, day_from, day_to, now=None):
    """Keeps finished Wialon Logistics routes of the period as trips (read-only on Wialon)."""
    now = int(now or time.time())
    key = (int(resource_id), day_from, day_to)
    with _sync_lock:
        if _synced.get(key, 0) > time.time() - FRESH_S:
            return
    ival_from = day_start(day_from) - BEFORE_S
    ival_to = min(day_start(day_to) + 86400 + AFTER_S, now + 86400)
    try:
        routes, orders = wialon.finished_routes(token, resource_id, ival_from, ival_to)
        unit_ids = {int((r.get("st") or {}).get("u") or 0) for r in routes} | {int(o.get("u") or 0) for o in orders}
        unit_ids.discard(0)
        rows = wialon_trip_rows(resource_id, routes, orders, day_from, day_to, _unit_names(token, unit_ids))
        store.insert_missing(rows)
        with _sync_lock:
            if len(_synced) > 200:
                _synced.clear()
            _synced[key] = time.time()
    except Exception as exc:
        print(f"[logistics] could not read routes from Wialon Logistics: {exc}")


# ---- Tracking and summaries ----


def _span(row):
    plan = row["plan"]
    return store.unix(row["window_from"]) - LEAD_S, max(store.unix(row["window_to"]), plan["returnAt"]) + TAIL_S


def tracking_for(token, row, now, keep=True):
    """Planned-vs-actual once the trip window has started; positions come from Wialon (read-only)."""
    t_from, t_to = _span(row)
    if now < t_from:
        return None
    pts = wialon.unit_track(token, row["unit_id"], t_from, t_to)
    tracking = track_plan(row["plan"], pts, now)
    if keep and now > t_to + SETTLE_S and not row.get("result"):
        store.save_result(row["id"], {**tracking, "path": []})
    return tracking


def to_summary(row, tracking, now):
    plan = row["plan"]
    w_from = store.unix(row["window_from"])
    w_to = store.unix(row["window_to"])
    stops = plan["stops"]
    return {
        "id": row["id"],
        "source": row["source"],
        "routeName": row["route_name"],
        "unitId": int(row["unit_id"]),
        "unitName": row["unit_name"],
        "warehouse": row.get("warehouse") or "",
        "day": row["day"],
        "windowFrom": w_from,
        "windowTo": w_to,
        "stops": len(stops),
        "km": plan["km"],
        "returnAt": plan["returnAt"],
        "weightKg": round(sum(s.get("weightKg", 0) for s in stops), 1),
        "amount": round(sum(s.get("amount", 0) for s in stops), 2),
        "createdBy": row.get("created_by") or "",
        "files": row.get("files") or "",
        "state": trip_state(w_from, max(w_to, plan["returnAt"]), now, tracking),
        "progress": progress_of(tracking) if tracking else None,
    }


SORTS = {
    "route": lambda s: s["routeName"].lower(),
    "vehicle": lambda s: s["unitName"].lower(),
    "window": lambda s: s["windowFrom"],
    "stops": lambda s: s["stops"],
    "km": lambda s: s["km"],
    "state": lambda s: TRIP_STATES.index(s["state"]),
}


def list_page(token, resource_id, day_from, day_to, q="", state=None, page=1, page_size=15, sort=None):
    """Trips of a period; states are first estimated from the clock, then measured for the visible page."""
    now = int(time.time())
    sync_wialon_history(token, resource_id, day_from, day_to, now)
    rows = store.list_trips(resource_id, day_from, day_to, with_results=True)
    q = (q or "").strip().lower()
    if q:
        rows = [r for r in rows if q in r["route_name"].lower() or q in r["unit_name"].lower()]
    rough = [to_summary(r, r.get("result"), now) for r in rows]
    counts = {k: 0 for k in TRIP_STATES}
    for s in rough:
        counts[s["state"]] += 1
    idx = [i for i, s in enumerate(rough) if not state or s["state"] == state]
    idx.sort(key=lambda i: (rough[i]["day"], TRIP_STATES.index(rough[i]["state"]), rough[i]["windowFrom"], rough[i]["routeName"]))
    if sort:
        col, _, direction = sort.partition(":")
        if col in SORTS:
            idx.sort(key=lambda i: SORTS[col](rough[i]), reverse=direction == "desc")
    page_idx = idx[(page - 1) * page_size : page * page_size]

    def summarise(i):
        row = rows[i]
        if row.get("result"):
            return to_summary(row, row["result"], now)
        try:
            return to_summary(row, tracking_for(token, row, now), now)
        except Exception as exc:
            print(f"[logistics] tracking failed: {exc}")
            return rough[i]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        summaries = list(pool.map(summarise, page_idx))
    return {"rows": summaries, "total": len(idx), "page": page, "pageSize": page_size, "counts": counts, "now": now, "tz": TZ_S}


def trip_detail(token, resource_id, trip_id):
    now = int(time.time())
    row = store.get_trip(resource_id, trip_id)
    tracking = tracking_for(token, row, now)
    return {**to_summary(row, tracking, now), "plan": row["plan"], "tracking": tracking, "now": now, "tz": TZ_S}


def report_trips(token, resource_id, day_from, day_to):
    """Every trip of a period with planned-vs-actual per stop; each vehicle's positions are read once per day."""
    now = int(time.time())
    sync_wialon_history(token, resource_id, day_from, day_to, now)
    rows = store.list_trips(resource_id, day_from, day_to, with_results=True)
    tracking = [r.get("result") for r in rows]
    groups = {}
    for i, r in enumerate(rows):
        if r.get("result"):
            continue
        t_from, t_to = _span(r)
        if now < t_from:
            continue
        g = groups.setdefault((r["unit_id"], r["day"]), {"unit": r["unit_id"], "from": t_from, "to": t_to, "idx": []})
        g["from"] = min(g["from"], t_from)
        g["to"] = max(g["to"], t_to)
        g["idx"].append(i)

    def run(g):
        try:
            pts = wialon.unit_track(token, g["unit"], g["from"], g["to"])
        except Exception as exc:
            print(f"[logistics] report tracking failed: {exc}")
            return
        for i in g["idx"]:
            r = rows[i]
            t_from, t_to = _span(r)
            tr = track_plan(r["plan"], [p for p in pts if t_from <= p[0] <= t_to], now)
            tracking[i] = tr
            if now > t_to + SETTLE_S:
                store.save_result(r["id"], {**tr, "path": []})

    with ThreadPoolExecutor(max_workers=REPORT_WORKERS) as pool:
        list(pool.map(run, groups.values()))
    return [(r, tracking[i], to_summary(r, tracking[i], now)) for i, r in enumerate(rows)]
