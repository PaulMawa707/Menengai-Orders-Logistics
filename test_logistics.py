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
        self.assertEqual(unit_track.call_count, 2)
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
             patch("services.logistics.plan._road_legs", side_effect=RuntimeError("offline")):
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
