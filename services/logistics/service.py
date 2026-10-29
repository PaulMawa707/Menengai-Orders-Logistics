import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from services import assets

from . import store, wialon
from .geo import decode_polyline, thin
from .plan import MAX_GEOMETRY, RULES, build_plan, road_legs
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
# Wialon order statuses: 2 completed in time, 3 completed overdue.
WIALON_DONE = {2: "on_time", 3: "late"}
VISITED = {"on_time", "early", "late"}
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
    left_depot = back_at_depot = None
    for o in pts:
        p = o.get("p") or {}
        r = p.get("r") or {}
        eta = int(r.get("vt") or o.get("tf") or 0)
        km += _num(r.get("m"))
        drive += _num(r.get("t"))
        done = int(o.get("s") or 0) in WIALON_DONE
        if int(o.get("f") or 0) & WAREHOUSE_FLAG:
            if depot is None:
                depot = {"name": o.get("n") or "Warehouse", "lat": float(o["y"]), "lon": float(o["x"])}
                if done:
                    left_depot = int(o.get("dtt") or o.get("stt") or 0) or None
            else:
                return_at = max(return_at, eta)
                if done:
                    back_at_depot = int(o.get("stt") or 0) or None
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
                **wialon_status(o),
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
    # Each order's path ("rp") is the planned road from the previous point to it.
    road = []
    for o in pts:
        leg = decode_polyline(o.get("rp") or "")
        road.extend(leg[1:] if road and leg else leg)
    if len(road) >= 2:
        geometry, routing = thin([[round(y, 5), round(x, 5)] for y, x in road], MAX_GEOMETRY), "wialon_road"
    else:
        ends = [[depot["lat"], depot["lon"]]] if depot else []
        geometry, routing = ends + [[s["lat"], s["lon"]] for s in stops] + ends, "wialon"
    return {
        "depot": depot,
        "start": start,
        "end": max(end, start),
        "returnAt": return_at,
        "km": round(km / 100) / 10,
        "driveMin": round(drive / 60),
        "serviceMin": round(service / 60),
        "stops": stops,
        "routing": routing,
        "geometry": geometry,
        "overrun": return_at > end,
        "rules": dict(RULES),
        "wStart": left_depot,
        "wReturn": back_at_depot,
    }


def wialon_status(order):
    """Wialon Logistics' own fulfilment of an order: status, arrival and departure."""
    s = int(order.get("s") or 0)
    if s not in WIALON_DONE:
        return {"ws": s}
    return {"ws": s, "wat": int(order.get("stt") or 0) or None, "wleft": int(order.get("dtt") or 0) or None}


def with_wialon(plan, tracking, now):
    """Planned-vs-actual with Wialon Logistics' fulfilment added: an order Wialon marks completed counts as visited
    even when the positions did not show it (Wialon uses each order's own radius)."""
    stops = plan.get("stops") or []
    if not any(s.get("ws") in WIALON_DONE for s in stops) and not plan.get("wStart") and not plan.get("wReturn"):
        return tracking
    if tracking is None:
        missed = now > plan["end"]
        tracking = {
            "at": now,
            "visits": [
                {"seq": s["seq"], "status": "missed" if missed else "pending", "arrivedAt": None, "leftAt": None, "closestM": None, "outOfOrder": False}
                for s in stops
            ],
            "startedAt": None,
            "returnedAt": None,
            "km": 0,
            "lastPos": None,
            "path": [],
        }
    by_seq = {s["seq"]: s for s in stops}
    visits = []
    for v in tracking["visits"]:
        s = by_seq.get(v["seq"], {})
        if s.get("ws") in WIALON_DONE:
            if v["status"] not in VISITED:
                v = {**v, "status": WIALON_DONE[s["ws"]], "arrivedAt": s.get("wat"), "leftAt": s.get("wleft"), "wialon": True}
            else:
                # In time or late as Wialon judged it (against the order's delivery interval); times stay from positions.
                v = {**v, "status": WIALON_DONE[s["ws"]]}
        visits.append(v)
    return {
        **tracking,
        "visits": visits,
        "startedAt": tracking.get("startedAt") or plan.get("wStart"),
        "returnedAt": tracking.get("returnedAt") or plan.get("wReturn"),
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
    """Keeps the Wialon Logistics routes of the period as trips: finished ones from history, and the ones still
    running (on the road, or left open with orders not visited). Read-only on Wialon."""
    now = int(now or time.time())
    key = (int(resource_id), day_from, day_to)
    with _sync_lock:
        if _synced.get(key, 0) > time.time() - FRESH_S:
            return
    ival_from = day_start(day_from) - BEFORE_S
    ival_to = min(day_start(day_to) + 86400 + AFTER_S, now + 86400)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finished = pool.submit(wialon.finished_routes, token, resource_id, ival_from, ival_to)
        running = pool.submit(wialon.active_routes, token, resource_id, day_start(day_from), day_start(day_to) + 86400)
        routes, orders, ok = [], [], 0
        for name, fut in (("history", finished), ("running routes", running)):
            try:
                r, o = fut.result()
                routes.extend(r)
                orders.extend(o)
                ok += 1
            except Exception as exc:
                print(f"[logistics] could not read {name} from Wialon Logistics: {exc}")
    if not ok:
        return
    try:
        unit_ids = {int((r.get("st") or {}).get("u") or 0) for r in routes} | {int(o.get("u") or 0) for o in orders}
        unit_ids.discard(0)
        rows = wialon_trip_rows(resource_id, routes, orders, day_from, day_to, _unit_names(token, unit_ids))
        store.insert_missing(rows)
        store.refresh_unfinished(rows)
        if ok == 2:
            with _sync_lock:
                if len(_synced) > 200:
                    _synced.clear()
                _synced[key] = time.time()
    except Exception as exc:
        print(f"[logistics] could not keep routes from Wialon Logistics: {exc}")


# ---- Tracking and summaries ----


def current_rules(rows):
    """Visit rules (stop radius, lateness) as configured now, also for trips planned before a change."""
    for r in rows:
        r["plan"] = {**r["plan"], "rules": {**r["plan"].get("rules", {}), **RULES}}
    return rows


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
    """Trips of a period, every one measured from positions and Wialon's fulfilment, so counts and filters are exact."""
    now = int(time.time())
    sync_wialon_history(token, resource_id, day_from, day_to, now)
    rows = current_rules(store.list_trips(resource_id, day_from, day_to, with_results=True))
    q = (q or "").strip().lower()
    if q:
        rows = [r for r in rows if q in r["route_name"].lower() or q in r["unit_name"].lower()]
    summaries = [to_summary(r, t, now) for r, t in zip(rows, track_all(token, rows, now))]
    counts = {k: 0 for k in TRIP_STATES}
    for s in summaries:
        counts[s["state"]] += 1
    idx = [i for i, s in enumerate(summaries) if not state or s["state"] == state]
    idx.sort(key=lambda i: (summaries[i]["day"], TRIP_STATES.index(summaries[i]["state"]), summaries[i]["windowFrom"], summaries[i]["routeName"]))
    if sort:
        col, _, direction = sort.partition(":")
        if col in SORTS:
            idx.sort(key=lambda i: SORTS[col](summaries[i]), reverse=direction == "desc")
    page_rows = [summaries[i] for i in idx[(page - 1) * page_size : page * page_size]]
    return {"rows": page_rows, "total": len(idx), "page": page, "pageSize": page_size, "counts": counts, "now": now, "tz": TZ_S}


def road_plan(row):
    """A Wialon route without a planned road path gets the road through its stops, in Wialon's order; kept once found."""
    plan = row["plan"]
    if plan.get("routing") != "wialon" or len(plan.get("stops") or []) < 1:
        return plan
    depot = plan.get("depot")
    pts = ([depot] if depot else []) + plan["stops"] + ([depot] if depot else [])
    try:
        _legs, geometry = road_legs([{"lat": p["lat"], "lon": p["lon"]} for p in pts])
    except Exception as exc:
        print(f"[logistics] could not route the planned road: {exc}")
        return plan
    if len(geometry) < 2:
        return plan
    road = {**plan, "geometry": thin([[round(y, 5), round(x, 5)] for y, x in geometry], MAX_GEOMETRY), "routing": "wialon_osrm"}
    try:
        store.update_plan(row["id"], road)
    except Exception as exc:
        print(f"[logistics] could not keep the planned road: {exc}")
    return road


def trip_detail(token, resource_id, trip_id):
    now = int(time.time())
    row = current_rules([store.get_trip(resource_id, trip_id)])[0]
    row["plan"] = road_plan(row)
    tracking = with_wialon(row["plan"], tracking_for(token, row, now), now)
    return {**to_summary(row, tracking, now), "plan": row["plan"], "tracking": tracking, "now": now, "tz": TZ_S}


def report_trips(token, resource_id, day_from, day_to, unit_id=None, failed=None):
    """Every trip of a period with planned-vs-actual per stop; each vehicle's positions are read once per day.

    Trips whose positions could not be loaded are appended to `failed` (when given).
    """
    now = int(time.time())
    sync_wialon_history(token, resource_id, day_from, day_to, now)
    rows = current_rules(store.list_trips(resource_id, day_from, day_to, with_results=True))
    if unit_id:
        rows = [r for r in rows if int(r["unit_id"]) == int(unit_id)]
    merged = track_all(token, rows, now, failed)
    return [(r, merged[i], to_summary(r, merged[i], now)) for i, r in enumerate(rows)]


def track_all(token, rows, now, failed=None):
    """Planned-vs-actual (with Wialon's fulfilment) of every trip; each vehicle's positions are read once per day."""
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
            if failed is not None:
                failed.extend(g["idx"])
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
    return [with_wialon(r["plan"], tracking[i], now) for i, r in enumerate(rows)]


# ---- Alerts ----

# A trip that has not left the warehouse this long after its planned start is reported.
NOT_STARTED_S = 30 * 60
ALERT_WINDOW_S = 24 * 3600
ALERTS_FRESH_S = 60
_alerts_cache = {}
_alerts_lock = threading.Lock()


def _hhmm(t):
    return datetime.fromtimestamp(int(t), tz=EAT).strftime("%H:%M")


def trip_alerts(row, tracking, s, now):
    """Events of one trip: stops visited (in time or late) or missed, trip completed, ended with missed stops, not started."""
    out = []
    plan = row["plan"]
    stops = {st["seq"]: st for st in plan["stops"]}
    unit, route, trip_id = s["unitName"], s["routeName"], row["id"]

    def add(key, kind, at, title, detail):
        out.append({"key": f"{key}:{trip_id}", "kind": kind, "at": int(at), "title": title, "detail": detail, "tripId": trip_id, "day": s["day"]})

    visits = (tracking or {}).get("visits", [])
    for v in visits:
        st = stops.get(v["seq"])
        if not st:
            continue
        where = f"{route} · stop {v['seq']} of {len(stops)}"
        if v["status"] in VISITED and v.get("arrivedAt"):
            dev = round((v["arrivedAt"] - st["eta"]) / 60)
            by = " · fulfilled in Wialon" if v.get("wialon") else ""
            if v["status"] == "late":
                add(f"visit:{v['seq']}", "late", v["arrivedAt"], f"{unit} reached {st['name']} late", f"{where} · arrived {_hhmm(v['arrivedAt'])}, {dev} min after the ETA{by}")
            else:
                add(f"visit:{v['seq']}", "visit", v["arrivedAt"], f"{unit} visited {st['name']}", f"{where} · arrived {_hhmm(v['arrivedAt'])}{by}")
        elif v["status"] == "missed":
            add(f"missed:{v['seq']}", "missed", plan["end"], f"{unit} missed {st['name']}", f"{where} · not visited by the end of the window ({_hhmm(plan['end'])})")

    p = s["progress"] or {}
    done = p.get("onTime", 0) + p.get("early", 0) + p.get("late", 0)
    if s["state"] == "completed":
        last = max([v.get("leftAt") or v.get("arrivedAt") or 0 for v in visits] + [0])
        at = (tracking or {}).get("returnedAt") or last or now
        add("completed", "completed", at, f"{unit} completed {route}", f"All {len(stops)} stops visited · {p.get('late', 0)} late")
    elif s["state"] == "ended":
        add("ended", "ended", max(s["windowTo"], s["returnAt"]), f"{unit} ended with missed stops", f"{route} · {done} of {len(stops)} visited · {p.get('missed', 0)} missed")
    elif (
        s["state"] == "in_progress"
        and not (tracking or {}).get("startedAt")
        and not done
        and now > plan["start"] + NOT_STARTED_S
    ):
        add("not_started", "not_started", plan["start"] + NOT_STARTED_S, f"{unit} has not left {plan['depot']['name'] if plan.get('depot') else 'the warehouse'}", f"{route} · planned start {_hhmm(plan['start'])}")
    return out


def alerts(token, resource_id):
    """Trip events of the last day, newest first (positions and Wialon fulfilment, read-only on Wialon)."""
    now = int(time.time())
    with _alerts_lock:
        hit = _alerts_cache.get(int(resource_id))
        if hit and hit[0] > time.time() - ALERTS_FRESH_S:
            return hit[1]
    today = local_day(now)
    yesterday = local_day(now - 86400)
    events = []
    for row, tracking, s in report_trips(token, resource_id, yesterday, today):
        if s["state"] == "scheduled":
            continue
        events.extend(e for e in trip_alerts(row, tracking, s, now) if now - ALERT_WINDOW_S <= e["at"] <= now)
    events.sort(key=lambda e: e["at"], reverse=True)
    out = {"alerts": events[:300], "now": now, "tz": TZ_S}
    with _alerts_lock:
        _alerts_cache[int(resource_id)] = (time.time(), out)
    return out


def to_report_trip(row, tracking, summary):
    """A trip with every stop planned against actual, for the Reports tab."""
    plan = row["plan"]
    visits = {v["seq"]: v for v in (tracking or {}).get("visits", [])}
    km = 0.0
    stops = []
    for st in plan["stops"]:
        km += st.get("legKm", 0)
        v = visits.get(st["seq"], {})
        arrived = v.get("arrivedAt")
        stops.append(
            {
                "seq": st["seq"],
                "name": st["name"],
                "location": st.get("location", ""),
                "invoice": st.get("invoice", ""),
                "amount": st.get("amount", 0),
                "weightKg": st.get("weightKg", 0),
                "lat": st["lat"],
                "lon": st["lon"],
                "eta": st["eta"],
                "etd": st["etd"],
                "plannedKm": round(km, 1),
                "status": v.get("status", "pending"),
                "arrivedAt": arrived,
                "leftAt": v.get("leftAt"),
                "deviationMin": round((arrived - st["eta"]) / 60) if arrived else None,
                "closestM": v.get("closestM"),
                "outOfOrder": bool(v.get("outOfOrder")),
            }
        )
    return {
        "id": row["id"],
        "source": summary["source"],
        "routeName": summary["routeName"],
        "unitId": summary["unitId"],
        "unitName": summary["unitName"],
        "warehouse": summary["warehouse"],
        "day": summary["day"],
        "state": summary["state"],
        "createdBy": summary["createdBy"],
        "planned": {
            "start": plan["start"],
            "end": plan["end"],
            "returnAt": plan["returnAt"],
            "km": plan["km"],
            "driveMin": plan.get("driveMin", 0),
            "serviceMin": plan.get("serviceMin", 0),
        },
        "actual": (
            {"startedAt": tracking.get("startedAt"), "returnedAt": tracking.get("returnedAt"), "km": tracking.get("km", 0)}
            if tracking
            else None
        ),
        "progress": summary["progress"],
        "stops": stops,
    }
