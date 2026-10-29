import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("APP_USERNAME", "test")
os.environ.setdefault("APP_PASSWORD", "test")
os.environ.setdefault("WIALON_TOKEN", "token")
os.environ.setdefault("WIALON_RESOURCE_ID", "25601229")

from services.logistics import service, store  # noqa: E402
from services.logistics.plan import build_plan  # noqa: E402
from services.logistics.track import progress_of, track_plan, trip_state  # noqa: E402

DEPOT = {"name": "MORL", "lat": -0.2880, "lon": 36.0449}
STOPS = [
    {"name": "A", "lat": -0.2970, "lon": 36.0497, "weightKg": 1000, "amount": 10},
    {"name": "B", "lat": -0.3100, "lon": 36.0700, "weightKg": 500, "amount": 5},
    {"name": "C", "lat": -0.3300, "lon": 36.1000, "weightKg": 250, "amount": 1},
]
START = 1_790_000_000
END = START + 12 * 3600


class PlanTests(unittest.TestCase):
    def test_etas_follow_travel_and_service_time(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        etas = [s["eta"] for s in plan["stops"]]
        self.assertGreater(etas[0], START)
        self.assertEqual(etas, sorted(etas))
        self.assertEqual(plan["stops"][1]["eta"] - plan["stops"][0]["etd"] > 0, True)
        self.assertGreater(plan["returnAt"], plan["stops"][-1]["etd"])
        self.assertEqual(plan["routing"], "straight")
        self.assertEqual(len(plan["geometry"]), len(STOPS) + 2)


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(STOPS, DEPOT, START, END, use_road=False)

    def test_visits_late_and_missed(self):
        a, b, _c = self.plan["stops"]
        pts = [
            (START - 60, DEPOT["lat"], DEPOT["lon"], 0),
            (START + 60, DEPOT["lat"] - 0.003, DEPOT["lon"], 40),
            (a["eta"], a["lat"], a["lon"], 0),
            (a["eta"] + 120, a["lat"], a["lon"], 0),
            (a["eta"] + 600, a["lat"] - 0.005, a["lon"], 40),
            # B is reached two hours after its ETA.
            (b["eta"] + 7200, b["lat"], b["lon"], 0),
            (b["eta"] + 7380, b["lat"], b["lon"], 0),
            (b["eta"] + 7400, b["lat"] - 0.005, b["lon"], 40),
            (END - 60, DEPOT["lat"], DEPOT["lon"], 0),
        ]
        tracking = track_plan(self.plan, pts, END + 60)
        status = [v["status"] for v in tracking["visits"]]
        self.assertEqual(status, ["on_time", "late", "missed"])
        self.assertEqual(tracking["startedAt"], START + 60)
        self.assertEqual(tracking["returnedAt"], END - 60)
        self.assertGreater(tracking["km"], 0)
        p = progress_of(tracking)
        self.assertEqual((p["onTime"], p["late"], p["missed"], p["pending"]), (1, 1, 1, 0))
        self.assertEqual(trip_state(START, END, END + 60, tracking), "ended")

    def test_vehicle_already_out_of_the_depot_starts_when_it_moves(self):
        a = self.plan["stops"][0]
        pts = [
            (START - 1800, -0.40, 36.20, 0),
            (START + 600, -0.40, 36.21, 50),
            (START + 4000, a["lat"], a["lon"], 0),
            (START + 4900, a["lat"] - 0.01, a["lon"], 40),
            # Passes through the depot later and leaves again; that is not the start of the trip.
            (START + 8000, DEPOT["lat"], DEPOT["lon"], 0),
            (START + 8400, DEPOT["lat"] - 0.01, DEPOT["lon"], 40),
        ]
        tracking = track_plan(self.plan, pts, START + 9000)
        self.assertEqual(tracking["startedAt"], START + 600)
        self.assertGreater(tracking["km"], 10)

    def test_driving_past_a_stop_is_not_a_visit(self):
        a = self.plan["stops"][0]
        pts = [
            # Drives through the circle at speed, then comes back and stops for 20 minutes.
            (a["eta"] - 600, a["lat"] + 0.001, a["lon"], 45),
            (a["eta"] - 570, a["lat"] - 0.001, a["lon"], 45),
            (a["eta"] - 400, a["lat"] - 0.01, a["lon"], 50),
            (a["eta"] + 300, a["lat"], a["lon"], 3),
            (a["eta"] + 1500, a["lat"], a["lon"], 0),
            (a["eta"] + 1560, a["lat"] - 0.01, a["lon"], 40),
        ]
        v = track_plan(self.plan, pts, END + 60)["visits"][0]
        self.assertEqual((v["arrivedAt"], v["leftAt"]), (a["eta"] + 300, a["eta"] + 1500))

    def test_a_parked_vehicle_reporting_rarely_counts_as_stopped(self):
        a = self.plan["stops"][0]
        pts = [(a["eta"], a["lat"], a["lon"], 0), (a["eta"] + 1800, a["lat"], a["lon"], 12), (a["eta"] + 1900, a["lat"] - 0.01, a["lon"], 40)]
        v = track_plan(self.plan, pts, END + 60)["visits"][0]
        self.assertEqual(v["status"], "on_time")
        self.assertEqual(v["leftAt"], a["eta"] + 1800)

    def test_a_move_in_and_out_of_the_yard_is_not_a_return(self):
        pts = [
            (START - 60, DEPOT["lat"], DEPOT["lon"], 0),
            (START + 60, DEPOT["lat"] - 0.005, DEPOT["lon"], 20),
            (START + 180, DEPOT["lat"], DEPOT["lon"], 0),
        ]
        tracking = track_plan(self.plan, pts, START + 600)
        self.assertEqual(tracking["startedAt"], START + 60)
        self.assertIsNone(tracking["returnedAt"])

    def test_states_from_the_clock(self):
        self.assertEqual(trip_state(START, END, START - 7200, None), "scheduled")
        self.assertEqual(trip_state(START, END, START + 60, None), "in_progress")
        self.assertEqual(trip_state(START, END, END + 60, None), "ended")


class WialonHistoryTests(unittest.TestCase):
    def test_finished_route_becomes_trip_without_warehouse_stops(self):
        rid = 1793027408357
        orders = [
            {"uid": 1, "n": "MORL", "f": 260, "tf": START, "tt": END, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": 27532305,
             "p": {"r": {"id": rid, "i": 0, "vt": START, "m": 0}}},
            {"uid": 2, "n": "Shop A", "f": 0, "tf": START, "tt": END, "y": -0.297, "x": 36.05, "u": 27532305,
             "p": {"a": "Kaptembwa", "ut": 600, "weight": "1320", "r": {"id": rid, "i": 1, "vt": START + 900, "m": 1121, "t": 300}}},
            {"uid": 3, "n": "MORL", "f": 264, "tf": START, "tt": END, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": 27532305,
             "p": {"r": {"id": rid, "i": 2, "vt": START + 3600, "m": 1500}}},
        ]
        routes = [{"uid": rid, "n": "ROUTE_1", "ord": [1, 2, 3], "st": {"u": 27532305}}]
        day = service.local_day(START)
        rows = service.wialon_trip_rows(25601229, routes, orders, day, day, {27532305: "VS - Menengai - KDE 999X"})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source_key"], f"wialon:25601229:{rid}")
        self.assertEqual(row["source"], "wialon")
        self.assertEqual(row["unit_name"], "VS - Menengai - KDE 999X")
        self.assertEqual([s["name"] for s in row["plan"]["stops"]], ["Shop A"])
        self.assertEqual(row["plan"]["depot"]["name"], "MORL")
        self.assertEqual(row["plan"]["returnAt"], START + 3600)
        self.assertEqual(row["plan"]["stops"][0]["weightKg"], 1320)

    def test_copied_route_uses_planned_visit_times_not_the_original_window(self):
        rid = 1792598340545
        later = START + 2 * 86400
        orders = [
            {"uid": 1, "n": "MORL", "f": 260, "tf": START, "tt": END, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": 7,
             "p": {"r": {"id": rid, "i": 0, "vt": later}}},
            {"uid": 2, "n": "Shop", "f": 64, "tf": START, "tt": END, "y": -0.297, "x": 36.05, "u": 7,
             "p": {"r": {"id": rid, "i": 1, "vt": later + 3600}}},
            {"uid": 3, "n": "MORL", "f": 264, "tf": START, "tt": END, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": 7,
             "p": {"r": {"id": rid, "i": 2, "vt": later + 7200}}},
        ]
        day = service.local_day(later)
        rows = service.wialon_trip_rows(25601229, [], orders, day, day, {7: "Truck"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["day"], day)
        self.assertEqual(rows[0]["plan"]["start"], later)
        self.assertEqual(rows[0]["plan"]["end"], later + 7200)

    def test_sync_keeps_running_and_finished_routes(self):
        def route(uid, unit, vt):
            orders = [
                {"uid": uid * 10 + 1, "n": "MORL", "f": 260, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": unit, "p": {"r": {"id": uid, "i": 0, "vt": vt}}},
                {"uid": uid * 10 + 2, "n": "Shop", "f": 0, "y": -0.297, "x": 36.05, "u": unit, "p": {"r": {"id": uid, "i": 1, "vt": vt + 3600}}},
            ]
            return {"uid": uid, "n": f"Route {uid}", "ord": [o["uid"] for o in orders], "st": {"u": unit}}, orders

        done, done_orders = route(111, 7, START)
        running, running_orders = route(222, 8, START + 1800)
        day = service.local_day(START)
        service._synced.clear()
        with patch("services.logistics.service.wialon.finished_routes", return_value=([done], done_orders)), \
             patch("services.logistics.service.wialon.active_routes", return_value=([running], running_orders)) as active, \
             patch("services.logistics.service._unit_names", return_value={7: "Truck 7", 8: "Truck 8"}), \
             patch("services.logistics.service.store.insert_missing") as insert, \
             patch("services.logistics.service.store.refresh_unfinished") as refresh:
            service.sync_wialon_history("token", 25601229, day, day, now=END)
        keys = sorted(r["source_key"] for r in insert.call_args.args[0])
        self.assertEqual(keys, ["wialon:25601229:111", "wialon:25601229:222"])
        refresh.assert_called_once()
        self.assertEqual(active.call_args.args[2], service.day_start(day))
        self.assertEqual(active.call_args.args[3], service.day_start(day) + 86400)

    def test_sync_still_keeps_running_routes_when_history_fails(self):
        day = service.local_day(START)
        service._synced.clear()
        orders = [{"uid": 1, "n": "Shop", "f": 0, "y": -0.297, "x": 36.05, "u": 8, "p": {"r": {"id": 5, "i": 0, "vt": START}}}]
        with patch("services.logistics.service.wialon.finished_routes", side_effect=RuntimeError("down")), \
             patch("services.logistics.service.wialon.active_routes", return_value=([{"uid": 5, "st": {"u": 8}}], orders)), \
             patch("services.logistics.service._unit_names", return_value={8: "Truck 8"}), \
             patch("services.logistics.service.store.insert_missing") as insert, \
             patch("services.logistics.service.store.refresh_unfinished"):
            service.sync_wialon_history("token", 25601229, day, day, now=END)
        self.assertEqual(len(insert.call_args.args[0]), 1)
        # Not marked as read, so the next request tries history again.
        self.assertEqual(service._synced, {})

    def test_refresh_only_patches_changed_wialon_trips(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        row = store.trip_row(resource_id=25601229, source="wialon", route_uid=9, route_name="R", unit_id=1, unit_name="T",
                             warehouse="MORL", day="2026-10-28", start=START, end=END, plan=plan)
        same = {"id": "a", "source_key": row["source_key"], "unit_id": 1, "unit_name": "T", "route_name": "R",
                "window_from": row["window_from"], "window_to": row["window_to"], "plan": plan}
        calls = []

        class Res:
            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        def fake(method, **kw):
            calls.append(method)
            return Res([same]) if method == "GET" else Res(None)

        with patch("services.logistics.store._request", side_effect=fake):
            store.refresh_unfinished([row])
            self.assertEqual(calls, ["GET"])
            same["route_name"] = "Old name"
            store.refresh_unfinished([row])
        self.assertEqual(calls, ["GET", "GET", "PATCH"])

    def test_finished_trip_only_gets_the_road_path(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        road = {**plan, "routing": "wialon_road", "geometry": [[-0.288, 36.045], [-0.29, 36.046], [-0.297, 36.05]]}
        row = store.trip_row(resource_id=25601229, source="wialon", route_uid=9, route_name="New name", unit_id=1, unit_name="T",
                             warehouse="MORL", day="2026-10-28", start=START, end=END, plan=road)
        have = {"id": "a", "source_key": row["source_key"], "unit_id": 1, "unit_name": "T", "route_name": "Old name",
                "window_from": row["window_from"], "window_to": row["window_to"], "plan": plan, "result_at": "2026-10-28T20:00:00Z"}
        patches = []

        class Res:
            def json(self):
                return [have]

        def fake(method, **kw):
            if method == "PATCH":
                patches.append(kw["json"])
            return Res()

        with patch("services.logistics.store._request", side_effect=fake):
            store.refresh_unfinished([row])
        self.assertEqual(list(patches[0]), ["plan"])
        self.assertEqual(patches[0]["plan"]["geometry"], road["geometry"])
        self.assertEqual(patches[0]["plan"]["stops"], plan["stops"])

    def test_opening_a_straight_wialon_trip_works_out_its_road_once(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        straight = {**plan, "routing": "wialon"}
        row = {"id": "t1", "plan": straight}
        road = [[-0.288, 36.045], [-0.29, 36.046], [-0.297, 36.05], [-0.31, 36.07]]
        with patch("services.logistics.service.road_legs", return_value=([], road)) as legs, \
             patch("services.logistics.service.store.update_plan") as keep:
            out = service.road_plan(row)
            self.assertEqual(out["routing"], "wialon_osrm")
            self.assertEqual(out["geometry"], road)
            keep.assert_called_once_with("t1", out)
            # Waypoints: warehouse, the stops in Wialon's order, warehouse.
            self.assertEqual(len(legs.call_args.args[0]), len(STOPS) + 2)
            self.assertIs(service.road_plan({"id": "t2", "plan": out}), out)
        self.assertEqual(legs.call_count, 1)

    def test_refresh_keeps_a_road_worked_out_for_a_wialon_trip(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        road = {**plan, "routing": "wialon_osrm", "geometry": [[-0.288, 36.045], [-0.3, 36.06], [-0.33, 36.1]]}
        row = store.trip_row(resource_id=25601229, source="wialon", route_uid=9, route_name="R", unit_id=1, unit_name="T",
                             warehouse="MORL", day="2026-10-28", start=START, end=END, plan={**plan, "routing": "wialon"})
        have = {"id": "a", "source_key": row["source_key"], "unit_id": 1, "unit_name": "T", "route_name": "R",
                "window_from": row["window_from"], "window_to": row["window_to"], "plan": road}
        methods = []

        class Res:
            def json(self):
                return [have]

        with patch("services.logistics.store._request", side_effect=lambda m, **kw: methods.append(m) or Res()):
            store.refresh_unfinished([row])
        self.assertEqual(methods, ["GET"])

    def test_wialon_route_uses_the_planned_road_path(self):
        from services.logistics.geo import decode_polyline

        rp = "bgw@}__{EQ~B[Cc@E_AGa@CQEC?IX]bAMVGPe@jACBGHE@EBG@@@?@@B@B?D@DAB?D?BABADCFCBCBA@"
        leg = decode_polyline(rp)
        self.assertAlmostEqual(leg[0][0], -0.28802, places=5)
        self.assertAlmostEqual(leg[0][1], 36.04495, places=5)
        orders = [
            {"uid": 1, "n": "MORL", "f": 260, "y": DEPOT["lat"], "x": DEPOT["lon"], "u": 7, "p": {"r": {"id": 5, "i": 0, "vt": START}}},
            {"uid": 2, "n": "Shop", "f": 0, "y": leg[-1][0], "x": leg[-1][1], "u": 7, "rp": rp, "p": {"r": {"id": 5, "i": 1, "vt": START + 600}}},
        ]
        day = service.local_day(START)
        plan = service.wialon_trip_rows(25601229, [], orders, day, day, {7: "T"})[0]["plan"]
        self.assertEqual(plan["routing"], "wialon_road")
        self.assertEqual(len(plan["geometry"]), len(leg))

    def test_upload_and_history_share_the_route_key(self):
        upload = store.trip_row(
            resource_id=25601229, source="upload", route_uid=1793000000, route_name="x", unit_id=1, unit_name="x",
            warehouse="MORL", day="2026-10-28", start=START, end=END, plan={},
        )
        self.assertEqual(upload["source_key"], "wialon:25601229:1793000000")


class ListTests(unittest.TestCase):
    def _row(self, trip_id, plan, result=None):
        return {
            "id": trip_id, "resource_id": 25601229, "source": "upload", "source_key": f"wialon:25601229:{trip_id}",
            "route_uid": 1, "route_name": f"Route {trip_id}", "unit_id": 27532302, "unit_name": "KDE 143E",
            "warehouse": "MORL", "day": service.local_day(plan["start"]),
            "window_from": store._iso(plan["start"]), "window_to": store._iso(plan["end"]),
            "plan": plan, "files": "", "created_by": "menengai", "result": result,
        }

    @patch("services.logistics.service.sync_wialon_history")
    @patch("services.logistics.service.wialon.unit_track", return_value=[])
    @patch("services.logistics.service.store.save_result")
    def test_counts_states_and_tracks_the_page(self, save_result, unit_track, _sync):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        rows = [self._row("a", plan), self._row("b", plan)]
        with patch("services.logistics.service.store.list_trips", return_value=rows), \
             patch("services.logistics.service.time.time", return_value=END + 5 * 3600):
            out = service.list_page("token", 25601229, rows[0]["day"], rows[0]["day"], page_size=5)
        self.assertEqual(out["total"], 2)
        self.assertEqual(out["counts"]["ended"], 2)
        self.assertEqual({r["state"] for r in out["rows"]}, {"ended"})
        self.assertTrue(all(r["progress"]["missed"] == 3 for r in out["rows"]))
        # Both trips are the same vehicle on the same day: its positions are read once.
        self.assertEqual(unit_track.call_count, 1)
        self.assertEqual(sum(out["counts"].values()), 2)
        # Finished and settled: results are kept so the list need not reload positions.
        self.assertEqual(save_result.call_count, 2)


class AppTests(unittest.TestCase):
    def setUp(self):
        from app import app

        self.client = app.test_client()
        with self.client.session_transaction() as s:
            s["logged_in"] = True
            s["username"] = "menengai"

    def test_trips_without_database(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SECRET_KEY": ""}):
            r = self.client.get("/api/logistics/trips?from=2026-10-28&to=2026-10-28")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["storage"])

    def test_rejects_long_ranges(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SECRET_KEY": "sb_secret_x"}):
            r = self.client.get("/api/logistics/trips?from=2026-01-01&to=2026-03-01")
        self.assertEqual(r.status_code, 400)

    def test_dispatch_records_the_trip(self):
        import app as app_module

        result = {
            "error": 0, "message": "Route created successfully", "planning_url": "u",
            "route_uid": 1793000000, "route_name": "VS - Menengai - KDE 143E - 2026-10-28 15:00",
            "ordered_stops": [{"name": "A", "location": "", "lat": -0.297, "lon": 36.05, "weightKg": 1000, "amount": 10, "invoice": ""}],
        }
        import pandas as pd

        df = pd.DataFrame([{"CUSTOMER NAME": "A", "LOCATION": "", "LAT": -0.297, "LONG": 36.05, "TONNAGE": 1, "AMOUNT": 10}])
        with patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SECRET_KEY": "sb_secret_x"}), \
             patch.object(app_module.optimized_orders, "process_multiple_excels", return_value=(df, "KDE143E")), \
             patch.object(app_module.optimized_orders, "send_orders_and_create_route", return_value=result), \
             patch.object(app_module.logistics, "record_upload", return_value="trip-1") as record:
            from io import BytesIO

            r = self.client.post(
                "/api/optimized/dispatch",
                data={
                    "orders": (BytesIO(b"x"), "KDE 143E.xlsx"),
                    "asset_item_id": "27532302",
                    "warehouse": "MORL",
                    "route_from": "2026-10-29T06:00",
                    "route_end": "2026-10-29T18:00",
                },
                content_type="multipart/form-data",
            )
        body = r.get_json()
        self.assertEqual(body["error"], 0, body)
        self.assertEqual(body["trip_id"], "trip-1")
        self.assertNotIn("ordered_stops", body)
        kwargs = record.call_args.kwargs
        self.assertEqual(kwargs["route_uid"], 1793000000)
        self.assertEqual(kwargs["unit_id"], 27532302)
        self.assertEqual(kwargs["files"], "KDE 143E.xlsx")
        self.assertEqual(body["trip_day"], "2026-10-29")

    def test_plan_preview_builds_the_schedule_without_dispatching(self):
        import app as app_module
        import pandas as pd
        from io import BytesIO

        df = pd.DataFrame([
            {"CUSTOMER NAME": "Far", "LOCATION": "", "LAT": -0.40, "LONG": 36.20, "TONNAGE": 1, "AMOUNT": 10},
            {"CUSTOMER NAME": "Near", "LOCATION": "", "LAT": -0.29, "LONG": 36.05, "TONNAGE": 2, "AMOUNT": 20},
        ])
        with patch.object(app_module.optimized_orders, "process_multiple_excels", return_value=(df, "KDE143E")), \
             patch.object(app_module.optimized_orders, "send_orders_and_create_route") as send, \
             patch("services.logistics.plan.road_legs", side_effect=RuntimeError("offline")):
            r = self.client.post(
                "/api/logistics/plan-preview",
                data={
                    "orders": (BytesIO(b"x"), "KDE 143E.xlsx"),
                    "asset_item_id": "27532302",
                    "warehouse": "MORL",
                    "stop_order": "warehouse",
                    "route_from": "2026-10-29T06:00",
                    "route_end": "2026-10-29T18:00",
                },
                content_type="multipart/form-data",
            )
        body = r.get_json()
        self.assertEqual(r.status_code, 200, body)
        send.assert_not_called()
        self.assertEqual([s["name"] for s in body["plan"]["stops"]], ["Near", "Far"])
        self.assertEqual(body["plan"]["depot"]["name"], "MORL")
        self.assertEqual(body["summary"]["delivery_points"], 2)
        self.assertGreater(body["plan"]["stops"][1]["eta"], body["plan"]["stops"][0]["eta"])

    def test_plan_preview_rejects_unknown_stop_order(self):
        from io import BytesIO

        r = self.client.post(
            "/api/logistics/plan-preview",
            data={"orders": (BytesIO(b"x"), "a.xlsx"), "asset_item_id": "27532302", "warehouse": "MORL",
                  "stop_order": "random", "route_from": "2026-10-29T06:00", "route_end": "2026-10-29T18:00"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)


class CountTests(unittest.TestCase):
    def test_counters_follow_the_measured_state_of_visible_trips(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        row = {
            "id": "a", "resource_id": 25601229, "source": "upload", "source_key": "wialon:25601229:1", "route_uid": 1,
            "route_name": "R", "unit_id": 1, "unit_name": "T", "warehouse": "MORL", "day": service.local_day(START),
            "window_from": store._iso(START), "window_to": store._iso(END), "plan": plan, "files": "", "created_by": "m", "result": None,
        }
        # Every stop visited during the window: the clock alone says on the road, positions say completed.
        pts = [(s["eta"], s["lat"], s["lon"], 0) for s in plan["stops"]]
        pts = [(p[0] + k, p[1], p[2], 0) for p in pts for k in (0, 150)]
        with patch("services.logistics.service.sync_wialon_history"), \
             patch("services.logistics.service.store.list_trips", return_value=[row]), \
             patch("services.logistics.service.wialon.unit_track", return_value=pts), \
             patch("services.logistics.service.store.save_result"), \
             patch("services.logistics.service.time.time", return_value=START + 4 * 3600):
            out = service.list_page("token", 25601229, row["day"], row["day"])
        self.assertEqual(out["rows"][0]["state"], "completed")
        self.assertEqual(out["counts"]["completed"], 1)
        self.assertEqual(out["counts"]["in_progress"], 0)


class ReportTests(unittest.TestCase):
    def setUp(self):
        from app import app

        self.client = app.test_client()
        with self.client.session_transaction() as s:
            s["logged_in"] = True
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        a, b, _c = plan["stops"]
        self.row = {
            "id": "t1", "source": "upload", "route_name": "VS - Menengai - KDE 143E", "unit_id": 27532302,
            "unit_name": "KDE 143E", "warehouse": "MORL", "day": service.local_day(START),
            "window_from": store._iso(START), "window_to": store._iso(END), "plan": plan, "files": "", "created_by": "m",
        }
        self.tracking = {
            "visits": [
                {"seq": 1, "status": "on_time", "arrivedAt": a["eta"] + 60, "leftAt": a["eta"] + 600, "closestM": 10, "outOfOrder": False},
                {"seq": 2, "status": "late", "arrivedAt": b["eta"] + 3600, "leftAt": None, "closestM": 5, "outOfOrder": False},
                {"seq": 3, "status": "missed", "arrivedAt": None, "leftAt": None, "closestM": 148, "outOfOrder": False},
            ],
            "startedAt": START + 300, "returnedAt": END - 600, "km": 42.5, "path": [],
        }
        self.summary = service.to_summary(self.row, self.tracking, END + 7200)

    def _get(self, url):
        with patch("services.logistics.service.report_trips", return_value=[(self.row, self.tracking, self.summary)]) as rt:
            r = self.client.get(url)
        return r, rt

    def test_report_data_has_planned_and_actual_per_order(self):
        r, rt = self._get(f"/api/logistics/report-data?from={self.row['day']}&to={self.row['day']}&unit=27532302")
        body = r.get_json()
        self.assertEqual(r.status_code, 200, body)
        self.assertEqual(rt.call_args.kwargs["unit_id"], 27532302)
        trip = body["trips"][0]
        self.assertEqual(trip["actual"], {"startedAt": START + 300, "returnedAt": END - 600, "km": 42.5})
        self.assertEqual([s["status"] for s in trip["stops"]], ["on_time", "late", "missed"])
        self.assertEqual(trip["stops"][0]["deviationMin"], 1)
        self.assertEqual(trip["stops"][1]["deviationMin"], 60)
        self.assertIsNone(trip["stops"][2]["arrivedAt"])
        self.assertEqual(trip["stops"][2]["closestM"], 148)

    def test_orders_filter_keeps_trips_with_a_matching_order(self):
        day = self.row["day"]
        r, _ = self._get(f"/api/logistics/report-data?from={day}&to={day}&orders=not_visited")
        self.assertEqual(len(r.get_json()["trips"]), 1)
        self.tracking["visits"][2]["status"] = "on_time"
        self.tracking["visits"][1]["status"] = "on_time"
        r, _ = self._get(f"/api/logistics/report-data?from={day}&to={day}&orders=not_visited")
        self.assertEqual(r.get_json()["trips"], [])

    def test_excel_routes_report_has_only_the_trips_sheet(self):
        import pandas as pd
        from io import BytesIO

        day = self.row["day"]
        r, _ = self._get(f"/api/logistics/report?from={day}&to={day}&kind=routes")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(list(pd.read_excel(BytesIO(r.data), sheet_name=None)), ["Trips"])
        r, _ = self._get(f"/api/logistics/report?from={day}&to={day}&kind=orders&orders=not_visited")
        sheets = pd.read_excel(BytesIO(r.data), sheet_name=None)
        self.assertEqual(sheets["Stops"]["Status"].tolist(), ["Missed"])

    def test_rejects_unknown_filter(self):
        r, _ = self._get(f"/api/logistics/report-data?from={self.row['day']}&to={self.row['day']}&orders=bogus")
        self.assertEqual(r.status_code, 400)


class WialonFulfilmentTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(STOPS, DEPOT, START, END, use_road=False)

    def _gps_missed_all(self):
        return track_plan(self.plan, [], END + 60)

    def test_orders_completed_in_wialon_count_as_visited(self):
        a, b, c = self.plan["stops"]
        a.update(ws=2, wat=a["eta"] + 120, wleft=a["eta"] + 600)
        b.update(ws=3, wat=b["eta"] + 7200, wleft=b["eta"] + 7300)
        c.update(ws=1)
        self.plan["wStart"], self.plan["wReturn"] = START + 300, END - 60
        t = service.with_wialon(self.plan, self._gps_missed_all(), END + 60)
        self.assertEqual([v["status"] for v in t["visits"]], ["on_time", "late", "missed"])
        self.assertTrue(t["visits"][0]["wialon"])
        self.assertEqual(t["visits"][0]["arrivedAt"], a["eta"] + 120)
        self.assertEqual((t["startedAt"], t["returnedAt"]), (START + 300, END - 60))

    def test_route_fulfilled_in_wialon_is_completed(self):
        for s in self.plan["stops"]:
            s.update(ws=2, wat=s["eta"], wleft=s["eta"] + 60)
        t = service.with_wialon(self.plan, self._gps_missed_all(), END + 60)
        self.assertEqual(trip_state(START, END, END + 60, t), "completed")
        # Even before positions were loaded.
        t = service.with_wialon(self.plan, None, END + 60)
        self.assertEqual(trip_state(START, END, END + 60, t), "completed")

    def test_gps_visit_keeps_its_times_with_wialons_verdict(self):
        a = self.plan["stops"][0]
        a.update(ws=2, wat=a["eta"] + 999)
        gps = self._gps_missed_all()
        gps["visits"][0].update(status="late", arrivedAt=a["eta"] + 5000)
        t = service.with_wialon(self.plan, gps, END + 60)
        self.assertEqual(t["visits"][0]["status"], "on_time")
        self.assertEqual(t["visits"][0]["arrivedAt"], a["eta"] + 5000)
        self.assertNotIn("wialon", t["visits"][0])

    def test_no_wialon_data_leaves_tracking_alone(self):
        self.assertIsNone(service.with_wialon(self.plan, None, END + 60))

    def test_uploads_take_wialon_status_by_position(self):
        wialon = {**self.plan, "stops": [{**s, "ws": 2, "wat": 111, "wleft": 222} for s in self.plan["stops"]], "wReturn": 333}
        merged = store.with_status(self.plan, wialon)
        self.assertEqual([s["ws"] for s in merged["stops"]], [2, 2, 2])
        self.assertEqual(merged["stops"][0]["eta"], self.plan["stops"][0]["eta"])
        self.assertEqual(merged["wReturn"], 333)


class AlertTests(unittest.TestCase):
    def test_trip_events(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        a, b, c = plan["stops"]
        row = {"id": "t1", "plan": plan, "route_name": "R1", "unit_id": 27532302, "unit_name": "VS - Menengai - KDE 143E", "source": "upload",
               "window_from": store._iso(START), "window_to": store._iso(END), "day": service.local_day(START)}
        tracking = {
            "visits": [
                {"seq": 1, "status": "on_time", "arrivedAt": a["eta"] + 60, "leftAt": None, "closestM": 5, "outOfOrder": False},
                {"seq": 2, "status": "late", "arrivedAt": b["eta"] + 3600, "leftAt": None, "closestM": 5, "outOfOrder": False, "wialon": True},
                {"seq": 3, "status": "missed", "arrivedAt": None, "leftAt": None, "closestM": 900, "outOfOrder": False},
            ],
            "startedAt": START + 100, "returnedAt": None, "km": 30, "lastPos": None, "path": [],
        }
        now = END + 7200
        s = service.to_summary(row, tracking, now)
        events = {e["kind"]: e for e in service.trip_alerts(row, tracking, s, now)}
        self.assertEqual(set(events), {"visit", "late", "missed", "ended"})
        self.assertIn("fulfilled in Wialon", events["late"]["detail"])
        self.assertEqual(events["missed"]["at"], END)
        self.assertTrue(all(e["tripId"] == "t1" for e in events.values()))

    def test_not_started_after_half_an_hour(self):
        plan = build_plan(STOPS, DEPOT, START, END, use_road=False)
        row = {"id": "t2", "plan": plan, "route_name": "R2", "unit_id": 1, "unit_name": "T", "source": "upload",
               "window_from": store._iso(START), "window_to": store._iso(END), "day": "2026-10-29"}
        now = START + 3600
        tracking = track_plan(plan, [], now)
        s = service.to_summary(row, tracking, now)
        kinds = [e["kind"] for e in service.trip_alerts(row, tracking, s, now)]
        self.assertEqual(kinds, ["not_started"])

    def test_alerts_endpoint(self):
        from app import app

        client = app.test_client()
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        payload = {"alerts": [{"key": "visit:1:t", "kind": "visit", "at": 1, "title": "x", "detail": "y", "tripId": "t", "day": "2026-10-29"}], "now": 2, "tz": 10800}
        with patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SECRET_KEY": "sb_secret_x"}), \
             patch("services.logistics.service.alerts", return_value=payload):
            r = client.get("/api/logistics/alerts")
        self.assertEqual(r.get_json()["alerts"][0]["kind"], "visit")


class RouteLabelTests(unittest.TestCase):
    def test_automatic_wialon_names_show_the_vehicle(self):
        self.assertEqual(service.route_label({"route_name": "ROUTE_1793025583811", "unit_name": "FG - Menengai - KCW 193E"}), "FG - Menengai - KCW 193E")
        self.assertEqual(service.route_label({"route_name": "VS - Menengai - KDE 143E - 2026-10-28 16:34", "unit_name": "x"}), "VS - Menengai - KDE 143E - 2026-10-28 16:34")


class MapTileTests(unittest.TestCase):
    def test_gurtam_tiles_for_the_wialon_user(self):
        from services.logistics import wialon

        wialon._tiles.clear()
        login = {"eid": "sid", "user": {"id": 27615568}, "gis_render": "", "base_url": "https://hst-api.wialon.com"}
        with patch("services.logistics.wialon.call", return_value=login) as call:
            url = wialon.map_tiles("token-12345678")
            self.assertEqual(url, "https://render-maps.wialon.com/hst-api.wialon.com/gis_render/{x}_{y}_{z}/27615568/tile.png")
            wialon.map_tiles("token-12345678")
        self.assertEqual(call.call_count, 1)


class AssetNameTests(unittest.TestCase):
    def test_catalog_has_full_wialon_names_and_still_matches_plates(self):
        from services import assets

        assets.clear_assets_cache()
        a = assets.find_asset_by_truck("KDE143E")
        self.assertEqual(a["item_id"], 27532302)
        self.assertEqual(a["name"], "VS - Menengai - KDE 143E")
        self.assertTrue(assets.get_asset_by_item_id(27519321)["name"].startswith("VS - Menengai"))
        fg = [x for x in assets.load_assets_catalog() if x["name"].startswith("FG - ")]
        self.assertTrue(fg)
        self.assertEqual(assets.find_asset_by_truck(fg[0]["plate"])["item_id"], fg[0]["item_id"])


class StopOrderTests(unittest.TestCase):
    def test_nearest_next_stop_chains_from_the_warehouse(self):
        import pandas as pd
        from wialon_core import order_stops

        # A line going east from the warehouse, plus one stop close to the warehouse on the far side.
        df = pd.DataFrame([
            {"CUSTOMER NAME": "East 2", "LAT": 0.0, "LONG": 0.20},
            {"CUSTOMER NAME": "West", "LAT": 0.0, "LONG": -0.11},
            {"CUSTOMER NAME": "East 1", "LAT": 0.0, "LONG": 0.10},
        ])
        by_warehouse = order_stops(df, 0.0, 0.0, "warehouse")["CUSTOMER NAME"].tolist()
        nearest = order_stops(df, 0.0, 0.0, "nearest")["CUSTOMER NAME"].tolist()
        self.assertEqual(by_warehouse, ["East 1", "West", "East 2"])
        self.assertEqual(nearest, ["East 1", "East 2", "West"])


if __name__ == "__main__":
    unittest.main()
