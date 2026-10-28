"""Planned stops against what the vehicle actually did, from its Wialon positions.

A position is (unix time, latitude, longitude, speed km/h).
"""

import math

from .geo import dist_m

# Positions this long before the window still count (vehicles often leave early).
LEAD_S = 3600
# And this long after it (late returns).
TAIL_S = 3 * 3600
MAX_PATH = 1500
# Ignore jumps faster than this when adding up distance (GPS glitches).
MAX_KMH = 200

TRIP_STATES = ["in_progress", "scheduled", "completed", "ended"]


def _accepts(seg, rules) -> bool:
    if not rules["completeOnStop"] and not rules["completeOnLeave"]:
        return True
    return (rules["completeOnStop"] and (seg["stopped"] or seg["to"] - seg["from"] >= 60)) or rules["completeOnLeave"]


def _first_visit(pts, lat, lon, after, rules):
    """First pass through the stop's circle at or after `after` that satisfies the visit rule."""
    r = rules["stopRadiusM"]
    # Rough bounding box so most points skip the distance calculation.
    d_lat = (r * 3) / 111_320
    d_lon = d_lat / max(0.1, math.cos(math.radians(lat)))
    closest = None
    seg = None
    for t, y, x, s in pts:
        if t < after:
            continue
        if abs(y - lat) > d_lat or abs(x - lon) > d_lon:
            if seg:
                if _accepts(seg, rules):
                    return seg, closest
                seg = None
            continue
        d = dist_m(lat, lon, y, x)
        if closest is None or d < closest:
            closest = d
        if d <= r:
            if not seg:
                seg = {"from": t, "to": t, "stopped": False, "open": True}
            seg["to"] = t
            if s <= rules["movingKmh"]:
                seg["stopped"] = True
        elif seg:
            seg["open"] = False
            if _accepts(seg, rules):
                return seg, closest
            seg = None
    # Still inside the radius at the latest position: the vehicle is at the stop now.
    return seg, closest


def track_plan(plan: dict, pts: list, now: int) -> dict:
    rules = plan["rules"]
    start = plan["start"] - LEAD_S
    visits = []
    latest = 0
    for st in plan["stops"]:
        seg, closest = _first_visit(pts, st["lat"], st["lon"], start, rules)
        status = "pending"
        if seg:
            if seg["from"] > st.get("dueBy", st["eta"] + rules["lateMin"] * 60):
                status = "late"
            elif rules["advanceMin"] > 0 and seg["from"] < st["eta"] - rules["advanceMin"] * 60:
                status = "early"
            else:
                status = "on_time"
        elif now > plan["end"]:
            status = "missed"
        out_of_order = bool(seg) and rules["strictSequence"] and seg["from"] < latest
        if seg:
            latest = max(latest, seg["from"])
        visits.append(
            {
                "seq": st["seq"],
                "status": status,
                "arrivedAt": seg["from"] if seg else None,
                "leftAt": seg["to"] if seg and not seg["open"] else None,
                "closestM": None if closest is None else round(closest),
                "outOfOrder": out_of_order,
            }
        )

    started_at = None
    returned_at = None
    depot = plan.get("depot")
    if depot:
        r = rules["stopRadiusM"]

        def in_depot(p):
            return dist_m(depot["lat"], depot["lon"], p[1], p[2]) <= r

        # Leaving the depot; a vehicle already away from it when the window opens starts when it first moves.
        first = next((k for k, p in enumerate(pts) if p[0] >= start), -1)
        if first >= 0 and not in_depot(pts[first]):
            i = next((k for k in range(first, len(pts)) if pts[k][3] > rules["movingKmh"]), -1)
        else:
            i = next((k for k in range(max(1, first), len(pts)) if not in_depot(pts[k]) and in_depot(pts[k - 1])), -1)
        if i >= 0:
            started_at = pts[i][0]
            last_visit = max([started_at] + [v["leftAt"] or v["arrivedAt"] or 0 for v in visits])
            for p in pts[i + 1 :]:
                if p[0] > last_visit and in_depot(p):
                    returned_at = p[0]
                    break
    else:
        started_at = next((p[0] for p in pts if p[0] >= start and p[3] > rules["movingKmh"]), None)

    until = returned_at if returned_at is not None else math.inf
    since = started_at if started_at is not None else start
    in_trip = [p for p in pts if since <= p[0] <= until]
    m = 0.0
    for a, b in zip(in_trip, in_trip[1:]):
        step = dist_m(a[1], a[2], b[1], b[2])
        dt = max(1, b[0] - a[0])
        if (step / dt) * 3.6 <= MAX_KMH:
            m += step
    every = max(1, math.ceil(len(in_trip) / MAX_PATH))
    path = [
        [round(p[1], 6), round(p[2], 6)]
        for k, p in enumerate(in_trip)
        if k % every == 0 or k == len(in_trip) - 1
    ]
    last = pts[-1] if pts else None
    return {
        "at": now,
        "visits": visits,
        "startedAt": started_at,
        "returnedAt": returned_at,
        "km": round(m / 100) / 10,
        "lastPos": {"t": last[0], "lat": last[1], "lon": last[2], "s": last[3]} if last else None,
        "path": path,
    }


def progress_of(tracking: dict) -> dict:
    p = {"total": len(tracking["visits"]), "onTime": 0, "late": 0, "early": 0, "missed": 0, "pending": 0, "outOfOrder": 0}
    key = {"on_time": "onTime", "late": "late", "early": "early", "missed": "missed"}
    for v in tracking["visits"]:
        p[key.get(v["status"], "pending")] += 1
        if v.get("outOfOrder"):
            p["outOfOrder"] += 1
    return p


def trip_state(window_from: int, window_to: int, now: int, tracking) -> str:
    if tracking:
        p = progress_of(tracking)
        if p["total"] and p["pending"] == 0 and p["missed"] == 0:
            return "completed"
    if now > window_to:
        return "ended"
    started = tracking and tracking.get("startedAt") and tracking["startedAt"] >= window_from - LEAD_S
    if now < window_from and not started:
        return "scheduled"
    return "in_progress"
