"""Wialon logistics core: Menengai Excel parsing + nearest-first optimized routing."""

import json
import re
import time
from datetime import datetime

import pandas as pd
import requests

from services.common import WAREHOUSES


def normalize_plate(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def extract_truck_number_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    patterns = [
        r"Truck\s*(?:Number|No\.?|#)?\s*[:\-]?\s*([A-Z0-9\- ]{4,})",
        r"\b([A-Z]{2,3}\s*\d{3,4}\s*[A-Z])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return normalize_plate(m.group(1))
    return None


def extract_coordinates(coord_str):
    try:
        if isinstance(coord_str, str) and ("LAT:" in coord_str and "LONG:" in coord_str):
            parts = coord_str.split("LONG:")
            latitude = float(parts[0].replace("LAT:", "").strip().replace(" ", ""))
            longitude = float(parts[1].strip().replace(" ", ""))
            return latitude, longitude
    except Exception:
        pass
    return None, None


def read_excel_to_df(excel_file):
    raw_df = pd.read_excel(excel_file, header=None)

    truck_number_norm = None
    if 0 in raw_df.columns:
        for row in raw_df[0].astype(str).tolist():
            truck_number_norm = extract_truck_number_from_text(row)
            if truck_number_norm:
                break

    header_row_idx = None
    for idx, row in raw_df.iterrows():
        if any(str(cell).strip().upper() == "NO." for cell in row):
            header_row_idx = idx
            break
    if header_row_idx is None:
        raise ValueError("Could not locate header row (cell 'NO.' not found).")

    df = pd.read_excel(excel_file, header=header_row_idx)
    df.columns = [
        re.sub(r"\s+", " ", str(col)).replace("\u00a0", " ").strip().upper()
        for col in df.columns
    ]
    df = df.loc[:, ~df.columns.str.startswith("UNNAMED")]

    required_cols = {"CUSTOMER ID", "CUSTOMER NAME", "LOCATION", "LOCATION COORDINATES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders Excel: {missing}")

    df = df[df["CUSTOMER ID"].notna()]
    df = df[~df["CUSTOMER NAME"].astype(str).str.contains("TOTAL", case=False, na=False)]

    for col in ("TONNAGE", "AMOUNT"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            ).fillna(0)
        else:
            df[col] = 0

    invoice_col = "INVOICE NO" if "INVOICE NO" in df.columns else None

    df_grouped = df.groupby(
        ["CUSTOMER ID", "CUSTOMER NAME", "LOCATION", "LOCATION COORDINATES", "REP"],
        as_index=False,
    ).agg(
        {
            "TONNAGE": "sum",
            "AMOUNT": "sum",
            **(
                {
                    invoice_col: lambda x: ", ".join(str(i) for i in x if pd.notna(i)),
                }
                if invoice_col
                else {}
            ),
        }
    )

    df_grouped[["LAT", "LONG"]] = df_grouped["LOCATION COORDINATES"].apply(
        lambda x: pd.Series(extract_coordinates(x))
    )
    df_grouped = df_grouped.dropna(subset=["LAT", "LONG"])
    return df_grouped, truck_number_norm


def process_multiple_excels(excel_files):
    all_gdfs = []
    truck_numbers = set()
    for excel_file in excel_files:
        gdf_joined, truck_number = read_excel_to_df(excel_file)
        if gdf_joined is not None and len(gdf_joined):
            all_gdfs.append(gdf_joined)
        if truck_number:
            truck_numbers.add(truck_number)

    if not all_gdfs:
        raise ValueError("No valid data found in any of the Excel files.")
    if len(truck_numbers) > 1:
        raise ValueError(
            f"Multiple truck numbers found (after normalization): {', '.join(sorted(truck_numbers))}"
        )

    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    combined_gdf = combined_gdf.drop_duplicates(subset=["CUSTOMER ID", "LOCATION"], keep="first")
    sole_truck = next(iter(truck_numbers)) if truck_numbers else None
    return combined_gdf, sole_truck


def _calc_dist(a, b):
    from math import atan2, cos, radians, sin, sqrt

    r = 6371
    y1, x1, y2, x2 = map(radians, [a["y"], a["x"], b["y"], b["x"]])
    dlat, dlon = y2 - y1, x2 - x1
    aa = sin(dlat / 2) ** 2 + cos(y1) * cos(y2) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(aa), (1 - aa) ** 0.5)


def _osrm_polyline(prev_coords, coords):
    try:
        osrm_url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{prev_coords['x']},{prev_coords['y']};{coords['x']},{coords['y']}"
            "?overview=full&geometries=polyline"
        )
        osrm_json = requests.get(osrm_url, timeout=15).json()
        if isinstance(osrm_json, dict) and osrm_json.get("routes"):
            return osrm_json["routes"][0].get("geometry")
    except Exception:
        pass
    return None


def send_orders_and_create_route(
    token, resource_id, unit_id, vehicle_name, df_grouped, tf, tt, warehouse_choice
):
    """Nearest-first route creation (TakaTaka optimized orders pattern)."""
    if warehouse_choice not in WAREHOUSES:
        return {"error": 1, "message": f"Unknown warehouse: {warehouse_choice}"}

    try:
        base_url = "https://hst-api.wialon.com/wialon/ajax.html"

        login_payload = {
            "svc": "token/login",
            "params": json.dumps({"token": str(token).strip()}),
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        login_response = requests.post(
            base_url, data=login_payload, headers=headers, timeout=30
        )
        login_result = login_response.json()

        if not isinstance(login_result, dict) or "eid" not in login_result:
            return {"error": 1, "message": f"Login failed: {login_result}"}
        session_id = login_result["eid"]

        wh = WAREHOUSES[warehouse_choice]
        wh_lat, wh_lon = wh["lat"], wh["lon"]
        wh_name = warehouse_choice
        wh_coords = f"{wh_lat}, {wh_lon}"

        df_grouped = df_grouped.copy()
        df_grouped["Distance_From_Warehouse"] = df_grouped.apply(
            lambda row: _calc_dist(
                {"y": wh_lat, "x": wh_lon},
                {"y": row["LAT"], "x": row["LONG"]},
            ),
            axis=1,
        )
        df_grouped = df_grouped.sort_values(
            "Distance_From_Warehouse", ascending=True
        ).reset_index(drop=True)

        orders = []
        for idx, row in df_grouped.iterrows():
            try:
                weight_kg = int(float(row.get("TONNAGE", 0)) * 1000)
            except Exception:
                weight_kg = 0
            coords = f"{row['LAT']}, {row['LONG']}"
            location = f"{row['LOCATION']} ({coords})"
            order_id = idx + 1
            orders.append(
                {
                    "y": float(row["LAT"]),
                    "x": float(row["LONG"]),
                    "tf": tf,
                    "tt": tt,
                    "n": row["CUSTOMER NAME"],
                    "f": 0,
                    "r": 100,
                    "id": order_id,
                    "p": {
                        "ut": 180,
                        "rep": True,
                        "w": weight_kg,
                        "v": 0,
                        "pr": order_id,
                        "criterions": {"max_late": 0, "use_unloading_late": 0},
                        "a": location,
                    },
                    "cmp": {"unitRequirements": {"values": []}},
                }
            )

        optimize_payload = {
            "svc": "order/optimize",
            "params": json.dumps(
                {
                    "itemId": int(resource_id),
                    "orders": orders,
                    "warehouses": [
                        {
                            "id": 0,
                            "y": wh_lat,
                            "x": wh_lon,
                            "n": wh_name,
                            "f": 260,
                            "a": f"{wh_name} ({wh_coords})",
                        },
                        {
                            "id": 99999,
                            "y": wh_lat,
                            "x": wh_lon,
                            "n": wh_name,
                            "f": 264,
                            "a": f"{wh_name} ({wh_coords})",
                        },
                    ],
                    "flags": 524419,
                    "units": [int(unit_id)],
                    "gis": {
                        "addPoints": 1,
                        "provider": 2,
                        "speed": 0,
                        "cityJams": 1,
                        "countryJams": 1,
                        "mode": "driving",
                        "departure_time": 1,
                        "avoid": [],
                        "traffic_model": "best_guess",
                    },
                    "priority": {},
                    "criterions": {"penalties_profile": "balanced"},
                    "pf": {
                        "n": wh_name,
                        "y": wh_lat,
                        "x": wh_lon,
                        "a": f"{wh_name} ({wh_coords})",
                    },
                    "pt": {
                        "n": wh_name,
                        "y": wh_lat,
                        "x": wh_lon,
                        "a": f"{wh_name} ({wh_coords})",
                    },
                    "tf": tf,
                    "tt": tt,
                }
            ),
            "sid": session_id,
        }

        optimize_response = requests.post(base_url, data=optimize_payload, timeout=60)
        optimize_result = optimize_response.json()

        route_summary = None
        end_warehouse_rp = None
        try:
            unit_key = str(unit_id)
            if isinstance(optimize_result, dict) and unit_key in optimize_result:
                unit_obj = optimize_result[unit_key]
                if unit_obj.get("routes"):
                    route_summary = unit_obj["routes"][0]
                if unit_obj.get("orders"):
                    for resp_order in reversed(unit_obj["orders"]):
                        if isinstance(resp_order, dict) and resp_order.get("f") == 264:
                            end_warehouse_rp = resp_order.get("rp") or resp_order.get("p")
                            break
        except Exception:
            pass

        route_orders = []
        current_time = int(time.time())
        route_id = current_time
        last_visit_time = int(tf)
        sequence_index = 0

        route_orders.append(
            {
                "uid": int(unit_id),
                "id": 0,
                "n": wh_name,
                "p": {
                    "ut": 0,
                    "rep": True,
                    "w": "0",
                    "c": "0",
                    "r": {
                        "vt": last_visit_time,
                        "ndt": 3,
                        "id": route_id,
                        "i": sequence_index,
                        "m": 0,
                        "t": 180,
                    },
                    "u": int(unit_id),
                    "a": f"{wh_name} ({wh_lat}, {wh_lon})",
                    "weight": "0",
                    "cost": "0",
                },
                "f": 260,
                "tf": tf,
                "tt": tt,
                "r": 100,
                "y": wh_lat,
                "x": wh_lon,
                "s": 0,
                "sf": 0,
                "trt": 0,
                "st": current_time,
                "cnm": 0,
                "ej": {},
                "cf": {},
                "cmp": {"unitRequirements": {"values": []}},
                "gfn": {"geofences": {}},
                "callMode": "create",
                "u": int(unit_id),
                "weight": "0",
                "cost": "0",
                "cargo": {"weight": "0", "cost": "0"},
            }
        )

        prev_coords = {"y": wh_lat, "x": wh_lon}

        for idx, cust_row in df_grouped.iterrows():
            order_id = idx + 1
            order_name = cust_row["CUSTOMER NAME"]
            coords = {"y": float(cust_row["LAT"]), "x": float(cust_row["LONG"])}

            weight_kg = int(float(cust_row.get("TONNAGE", 0)) * 1000)
            cost_val = float(cust_row.get("AMOUNT", 0.0))
            location = f"{cust_row['LOCATION']} ({coords['y']}, {coords['x']})"

            order_tm = max(last_visit_time + 180, int(tf))
            mileage = int(_calc_dist(prev_coords, coords) * 1000)
            order_rp = _osrm_polyline(prev_coords, coords)

            sequence_index += 1
            route_orders.append(
                {
                    "uid": int(unit_id),
                    "id": order_id,
                    "n": order_name,
                    "p": {
                        "ut": 180,
                        "rep": True,
                        "w": str(weight_kg),
                        "c": str(int(cost_val)),
                        "r": {
                            "vt": order_tm,
                            "ndt": 3,
                            "id": route_id,
                            "i": sequence_index,
                            "m": mileage,
                            "t": 0,
                        },
                        "u": int(unit_id),
                        "a": location,
                        "weight": str(weight_kg),
                        "cost": str(int(cost_val)),
                    },
                    "f": 0,
                    "tf": tf,
                    "tt": tt,
                    "r": 100,
                    "y": coords["y"],
                    "x": coords["x"],
                    "s": 0,
                    "sf": 0,
                    "trt": 0,
                    "st": current_time,
                    "cnm": 0,
                    **({"rp": order_rp} if order_rp else {}),
                    "ej": {},
                    "cf": {},
                    "cmp": {"unitRequirements": {"values": []}},
                    "gfn": {"geofences": {}},
                    "callMode": "create",
                    "u": int(unit_id),
                    "weight": str(weight_kg),
                    "cost": str(int(cost_val)),
                    "cargo": {"weight": str(weight_kg), "cost": str(int(cost_val))},
                }
            )

            prev_coords = coords
            last_visit_time = order_tm

        mileage_back = int(
            _calc_dist(prev_coords, {"y": wh_lat, "x": wh_lon}) * 1000
        )
        final_id = max([o.get("id", 0) for o in route_orders]) + 1
        sequence_index += 1

        if not end_warehouse_rp:
            end_warehouse_rp = _osrm_polyline(prev_coords, {"y": wh_lat, "x": wh_lon})

        route_orders.append(
            {
                "uid": int(unit_id),
                "id": final_id,
                "n": wh_name,
                "p": {
                    "ut": 0,
                    "rep": True,
                    "w": "0",
                    "c": "0",
                    "r": {
                        "vt": last_visit_time + 180,
                        "ndt": 3,
                        "id": route_id,
                        "i": sequence_index,
                        "m": mileage_back,
                        "t": 180,
                    },
                    "u": int(unit_id),
                    "a": f"{wh_name} ({wh_lat}, {wh_lon})",
                    "weight": "0",
                    "cost": "0",
                },
                "f": 264,
                "tf": tf,
                "tt": tt,
                "r": 100,
                "y": wh_lat,
                "x": wh_lon,
                "s": 0,
                "sf": 0,
                "trt": 0,
                "st": current_time,
                "cnm": 0,
                **({"rp": end_warehouse_rp} if end_warehouse_rp else {}),
                "ej": {},
                "cf": {},
                "cmp": {"unitRequirements": {"values": []}},
                "gfn": {"geofences": {}},
                "callMode": "create",
                "u": int(unit_id),
                "weight": "0",
                "cost": "0",
                "cargo": {"weight": "0", "cost": "0"},
            }
        )

        total_mileage = sum(order["p"]["r"]["m"] for order in route_orders)
        total_cost = sum(
            float(order["p"]["c"]) for order in route_orders if order["f"] == 0
        )
        total_weight = sum(
            int(order["p"]["w"]) for order in route_orders if order["f"] == 0
        )

        batch_payload = {
            "svc": "core/batch",
            "params": json.dumps(
                {
                    "params": [
                        {
                            "svc": "order/route_update",
                            "params": {
                                "itemId": int(resource_id),
                                "orders": route_orders,
                                "uid": route_id,
                                "callMode": "create",
                                "exp": 0,
                                "f": 0,
                                "n": f"{vehicle_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                "summary": {
                                    "countOrders": len(route_orders),
                                    "duration": (
                                        route_summary.get("duration", 0)
                                        if isinstance(route_summary, dict)
                                        else 0
                                    ),
                                    "mileage": total_mileage,
                                    "priceMileage": float(total_mileage) / 1000,
                                    "priceTotal": total_cost,
                                    "weight": total_weight,
                                    "cost": total_cost,
                                },
                            },
                        }
                    ],
                    "flags": 0,
                }
            ),
            "sid": session_id,
        }

        route_response = requests.post(base_url, data=batch_payload, timeout=60)
        route_result = route_response.json()

        planning_url = (
            f"https://apps.wialon.com/logistics/?lang=en&sid={session_id}#/distrib/step3"
        )

        if isinstance(route_result, list):
            has_error = any(
                isinstance(item, dict) and item.get("error", 0) != 0
                for item in route_result
            )
            if not has_error:
                return {
                    "error": 0,
                    "message": "Route created successfully",
                    "planning_url": planning_url,
                    "optimize_result": optimize_result,
                    "route_result": route_result,
                }
            error_item = next(
                (
                    item
                    for item in route_result
                    if isinstance(item, dict) and item.get("error", 0) != 0
                ),
                None,
            )
            return {
                "error": (error_item or {}).get("error", 1),
                "message": (error_item or {}).get(
                    "reason", "Unknown error in batch response"
                ),
            }

        if isinstance(route_result, dict) and route_result.get("error", 1) == 0:
            return {
                "error": 0,
                "message": "Route created successfully",
                "planning_url": planning_url,
                "optimize_result": optimize_result,
                "route_result": route_result,
            }

        return {"error": 1, "message": f"Unexpected or error response: {route_result}"}

    except Exception as e:
        return {"error": 1, "message": f"An unexpected error occurred: {str(e)}"}
