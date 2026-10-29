import os

import requests

from .geo import dist_m

OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org").rstrip("/")
FALLBACK_SPEED_KMH = 30
SERVICE_MIN = float(os.getenv("LOGISTICS_SERVICE_MIN", "15"))
# OSRM limits waypoints per request; longer routes are split into overlapping parts.
CHUNK = 80

RULES = {
    "stopRadiusM": int(os.getenv("LOGISTICS_STOP_RADIUS_M", "100")),
    "lateMin": int(os.getenv("LOGISTICS_LATE_MIN", "15")),
    "advanceMin": 0,
    "strictSequence": False,
    "completeOnStop": True,
    "completeOnLeave": True,
    "movingKmh": 5,
}


def _osrm_part(pts):
    coords = ";".join(f"{p['lon']:.6f},{p['lat']:.6f}" for p in pts)
    url = f"{OSRM_URL}/route/v1/driving/{coords}?overview=simplified&geometries=geojson&steps=false"
    res = requests.get(url, timeout=15, headers={"user-agent": "Menengai-Orders-Logistics"})
    res.raise_for_status()
    js = res.json()
    route = (js.get("routes") or [None])[0]
    if js.get("code") != "Ok" or not route:
        raise RuntimeError(f"OSRM {js.get('code')}")
    legs = [{"m": leg["distance"], "s": leg["duration"]} for leg in route["legs"]]
    geometry = [[y, x] for x, y in route["geometry"]["coordinates"]]
    return legs, geometry


def road_legs(pts):
    legs, geometry = [], []
    for i in range(0, len(pts) - 1, CHUNK - 1):
        part_legs, part_geom = _osrm_part(pts[i : i + CHUNK])
        legs.extend(part_legs)
        geometry.extend(part_geom[1:] if geometry else part_geom)
    return legs, geometry


def _straight_legs(pts):
    legs = []
    for a, b in zip(pts, pts[1:]):
        m = dist_m(a["lat"], a["lon"], b["lat"], b["lon"])
        legs.append({"m": m, "s": m / (FALLBACK_SPEED_KMH / 3.6)})
    return legs


def build_plan(stops, depot, start: int, end: int, use_road=True) -> dict:
    """Planned times for every stop from the window start: travel time, then the stop's service time.

    `stops` are in driving order: dicts with name, location, lat, lon, amount, weightKg, invoice.
    `depot` is {name, lat, lon} or None; with a depot the trip starts and ends there.
    """
    pts = ([depot] if depot else []) + list(stops) + ([depot] if depot else [])
    legs, geometry, routing = [], None, "straight"
    if len(pts) >= 2:
        if use_road:
            try:
                road, geom = road_legs(pts)
                if len(road) == len(pts) - 1:
                    legs, geometry, routing = road, geom, "osrm"
            except Exception as exc:
                print(f"[logistics] OSRM failed, using straight lines: {exc}")
        if routing == "straight":
            legs = _straight_legs(pts)

    t = float(start)
    li = 0
    km = drive = service = 0.0
    out = []
    for i, st in enumerate(stops):
        # Without a depot the first stop is the starting point.
        leg = None
        if depot or out:
            leg = legs[li] if li < len(legs) else None
            li += 1
        leg_m = leg["m"] if leg else 0.0
        t += leg["s"] if leg else 0.0
        km += leg_m
        drive += leg["s"] if leg else 0.0
        svc = SERVICE_MIN * 60
        out.append(
            {
                "seq": i + 1,
                "name": str(st.get("name", "")).strip() or f"Stop {i + 1}",
                "location": str(st.get("location", "") or ""),
                "lat": float(st["lat"]),
                "lon": float(st["lon"]),
                "amount": round(float(st.get("amount", 0) or 0), 2),
                "weightKg": round(float(st.get("weightKg", 0) or 0), 1),
                "invoice": str(st.get("invoice", "") or ""),
                "legKm": round(leg_m / 10) / 100,
                "eta": round(t),
                "etd": round(t + svc),
            }
        )
        t += svc
        service += svc
    if depot and li < len(legs):
        t += legs[li]["s"]
        km += legs[li]["m"]
        drive += legs[li]["s"]

    return {
        "depot": {"name": depot["name"], "lat": depot["lat"], "lon": depot["lon"]} if depot else None,
        "start": int(start),
        "end": int(end),
        "returnAt": round(t),
        "km": round(km / 100) / 10,
        "driveMin": round(drive / 60),
        "serviceMin": round(service / 60),
        "stops": out,
        "routing": routing,
        "geometry": geometry or [[p["lat"], p["lon"]] for p in pts],
        "overrun": t > end,
        "rules": dict(RULES),
    }
