from io import BytesIO
import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from wialon_core import (
    _extract_wialon_error,
    _unique_id_base,
    process_multiple_excels,
    send_orders_and_create_route,
)


def _orders_workbook(tonnage, amount, invoice):
    rows = [
        ["Truck Number: KDM 138L", None, None, None, None, None, None, None, None],
        [
            "NO.",
            "CUSTOMER ID",
            "CUSTOMER NAME",
            "LOCATION",
            "LOCATION COORDINATES",
            "REP",
            "TONNAGE",
            "AMOUNT",
            "INVOICE NO.",
        ],
        [
            1,
            "C1",
            "Customer One",
            "Nairobi",
            "LAT: -1.2 LONG: 36.8",
            "Rep",
            tonnage,
            amount,
            invoice,
        ],
    ]
    stream = BytesIO()
    pd.DataFrame(rows).to_excel(stream, index=False, header=False)
    stream.seek(0)
    return stream


class MultipleExcelTests(unittest.TestCase):
    def test_combines_duplicate_delivery_points_before_routing(self):
        combined, truck = process_multiple_excels(
            [
                _orders_workbook(1, 100, "INV-A"),
                _orders_workbook(2, 200, "INV-B"),
            ]
        )

        self.assertEqual(truck, "KDM138L")
        self.assertEqual(len(combined), 1)
        self.assertEqual(float(combined.iloc[0]["TONNAGE"]), 3)
        self.assertEqual(float(combined.iloc[0]["AMOUNT"]), 300)
        self.assertEqual(combined.iloc[0]["INVOICE NO"], "INV-A, INV-B")


class WialonRoutePayloadTests(unittest.TestCase):
    def test_unique_id_base_is_never_zero(self):
        self.assertGreater(_unique_id_base(1_784_275_705), 0)
        self.assertEqual(_unique_id_base(0), 1)

    def test_extracts_unknown_order_get_error(self):
        error = _extract_wialon_error(
            [{"error": 4, "reason": "UNKNOWN_ORDER_GET_ERROR #0 id=0"}]
        )
        self.assertEqual(error[0], 4)
        self.assertIn("UNKNOWN_ORDER_GET_ERROR", error[1])

    @patch("wialon_core._osrm_polyline", return_value=None)
    @patch("wialon_core.requests.post")
    def test_route_update_uses_nonzero_order_ids_and_route_id(self, post, _polyline):
        login = Mock()
        login.json.return_value = {"eid": "sid-1"}
        optimize = Mock()
        optimize.json.return_value = {}
        create = Mock()
        create.json.return_value = {"orders": [{"id": 11, "callMode": "create"}]}
        post.side_effect = [login, optimize, create]

        df = pd.DataFrame(
            [
                {
                    "CUSTOMER NAME": "Customer One",
                    "LOCATION": "Nairobi",
                    "TONNAGE": 1,
                    "AMOUNT": 100,
                    "LAT": -1.2,
                    "LONG": 36.8,
                }
            ]
        )

        result = send_orders_and_create_route(
            "token", 25601229, 27513423, "KDM 125L", df, 100, 200, "MORL"
        )

        self.assertEqual(result["error"], 0)
        batch = json.loads(post.call_args_list[2].kwargs["data"]["params"])
        route_params = batch["params"][0]["params"]
        order_ids = [order["id"] for order in route_params["orders"]]

        self.assertEqual(batch["params"][0]["svc"], "order/route_update")
        self.assertIn("routeId", route_params)
        self.assertNotIn("uid", route_params)
        self.assertGreater(min(order_ids), 0)
        self.assertEqual(len(order_ids), len(set(order_ids)))
        self.assertTrue(all("uid" not in order for order in route_params["orders"]))
        self.assertTrue(all(order["u"] == 27513423 for order in route_params["orders"]))
        self.assertEqual(result["route_uid"], route_params["routeId"])
        self.assertEqual(result["ordered_stops"][0]["name"], "Customer One")
        self.assertEqual(result["ordered_stops"][0]["weightKg"], 1000)
        self.assertEqual(result["ordered_stops"][0]["invoice"], "")


if __name__ == "__main__":
    unittest.main()
