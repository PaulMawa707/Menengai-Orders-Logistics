"""Read-only Wialon calls for the Trips dashboard (positions, finished Logistics routes, unit names)."""

import json
import threading
import time

import requests

BASE_URL = "https://hst-api.wialon.com/wialon/ajax.html"
# Wialon sessions close after about 5 minutes without requests.
SESSION_TTL_S = 240


class WialonError(RuntimeError):
    def __init__(self, svc, code, reason=""):
        super().__init__(f"Wialon {svc}: error {code}{f' - {reason}' if reason else ''}")
        self.code = code


def call(svc, params, sid=None, timeout=60):
    data = {"svc": svc, "params": json.dumps(params or {})}
    if sid:
        data["sid"] = sid
    res = requests.post(BASE_URL, data=data, timeout=timeout)
    res.raise_for_status()
    out = res.json()
    if isinstance(out, dict) and out.get("error"):
        raise WialonError(svc, out["error"], out.get("reason", ""))
    return out


# Each thread keeps its own session: loaded messages live in a per-session buffer.
_local = threading.local()


def session(token: str) -> str:
    s = getattr(_local, "s", None)
    if s and s["token"] == token and s["exp"] > time.time():
        s["exp"] = time.time() + SESSION_TTL_S
        return s["sid"]
    sid = call("token/login", {"token": token}, timeout=30)["eid"]
    _local.s = {"token": token, "sid": sid, "exp": time.time() + SESSION_TTL_S}
    return sid


def _with_session(token, fn):
    try:
        return fn(session(token))
    except WialonError as exc:
        if exc.code != 1:
            raise
        _local.s = None
        return fn(session(token))


_track_cache = {}
_track_lock = threading.Lock()


def _load_track(token, unit_id, t_from, t_to):
    def run(sid):
        try:
            r = call(
                "messages/load_interval",
                {"itemId": int(unit_id), "timeFrom": int(t_from), "timeTo": int(t_to), "flags": 0x1, "flagsMask": 0xFF01, "loadCount": 0xFFFFFFFF},
                sid,
            )
        except WialonError as exc:
            if exc.code == 1001:
                return []
            raise
        finally:
            try:
                call("messages/unload", {}, sid, timeout=15)
            except Exception:
                pass
        pts = []
        for m in r.get("messages") or []:
            pos = m.get("pos")
            if pos and (pos.get("y") or pos.get("x")):
                pts.append((m["t"], pos["y"], pos["x"], pos.get("s") or 0))
        pts.sort(key=lambda p: p[0])
        return pts

    return _with_session(token, run)


def unit_track(token, unit_id, t_from, t_to):
    """Positions of a vehicle; recent intervals refresh every minute, finished ones are kept longer."""
    now = int(time.time())
    end = min(int(t_to), now)
    if end <= t_from:
        return []
    done = t_to < now - 600
    key = (token[-8:], int(unit_id), int(t_from), int(t_to))
    with _track_lock:
        hit = _track_cache.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
    pts = _load_track(token, unit_id, t_from, end)
    with _track_lock:
        if len(_track_cache) > 300:
            _track_cache.clear()
        _track_cache[key] = (time.time() + (1800 if done else 60), pts)
    return pts


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return list(v.values())
    return []


def finished_routes(token, resource_id, ival_from, ival_to):
    """Routes and their orders from Wialon Logistics history (finished routes only)."""

    def run(sid):
        out = call(
            "core/batch",
            {
                "params": [
                    {"svc": "order/get_routes_history", "params": {"itemId": int(resource_id), "ivalType": 1, "ivalFrom": int(ival_from), "ivalTo": int(ival_to)}},
                    {"svc": "order/get_orders_history", "params": {"itemId": int(resource_id), "ivalType": 1, "ivalFrom": int(ival_from), "ivalTo": int(ival_to)}},
                ],
                "flags": 0,
            },
            sid,
            timeout=120,
        )
        hist, ohist = (r if isinstance(r, dict) and not r.get("error") else {} for r in out)
        return _as_list(hist.get("routes")), _as_list(ohist.get("orders"))

    return _with_session(token, run)


LOGISTICS_ROUTES_URL = "https://logistics.wialon.com/api/routes"


def active_routes(token, resource_id, t_from, t_to):
    """Routes running in Wialon Logistics in a period (on the road, or left open with orders not visited), with their orders.

    The whole resource's active orders are far too large to load through the remote API; this Logistics service
    returns only the routes of the period.
    """
    res = requests.get(
        LOGISTICS_ROUTES_URL,
        params={"resourceId": int(resource_id), "token": token, "from": int(t_from), "to": int(t_to)},
        timeout=120,
    )
    res.raise_for_status()
    data = res.json()
    if isinstance(data, dict) and data.get("error"):
        raise WialonError("logistics/api/routes", data["error"], data.get("detail", ""))
    routes = _as_list(data.get("routes") if isinstance(data, dict) else data)
    orders = [o for r in routes for o in _as_list(r.get("orders"))]
    return routes, orders


def unit_names(token, unit_ids):
    ids = sorted({int(u) for u in unit_ids if u})
    if not ids:
        return {}

    def run(sid):
        out = call(
            "core/batch",
            {"params": [{"svc": "core/search_item", "params": {"id": u, "flags": 1}} for u in ids], "flags": 0},
            sid,
        )
        names = {}
        for u, r in zip(ids, out):
            item = r.get("item") if isinstance(r, dict) else None
            if item and item.get("nm"):
                names[u] = item["nm"]
        return names

    return _with_session(token, run)
