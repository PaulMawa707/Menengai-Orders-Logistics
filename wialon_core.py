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
    expected_header_cells = {"CUSTOMER ID", "CUSTOMER NAME", "LOCATION"}
    for idx, row in raw_df.iterrows():
        normalized_cells = {
            re.sub(r"\s+", " ", str(cell)).replace("\u00a0", " ").strip().upper()
            for cell in row
            if pd.notna(cell)
        }
        if "NO." in normalized_cells and expected_header_cells.issubset(normalized_cells):
            header_row_idx = idx
            break
    if header_row_idx is None:
        raise ValueError("Could not locate the orders table header row.")

    df = pd.read_excel(excel_file, header=header_row_idx)
    df.columns = [
        re.sub(r"\s+", " ", str(col)).replace("\u00a0", " ").strip().upper()
        for col in df.columns
    ]
    df = df.loc[:, ~df.columns.str.startswith("UNNAMED")]

    if "LOCATION COORDINATES" not in df.columns and "COORDINATES" in df.columns:
        df = df.rename(columns={"COORDINATES": "LOCATION COORDINATES"})
    if "INVOICE NO" not in df.columns and "INVOICE NO." in df.columns:
        df = df.rename(columns={"INVOICE NO.": "INVOICE NO"})

    required_cols = {"CUSTOMER ID", "CUSTOMER NAME", "LOCATION", "LOCATION COORDINATES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders Excel: {missing}")

    df = df[df["CUSTOMER ID"].notna()]
    df = df[~df["CUSTOMER NAME"].astype(str).str.contains("TOTAL", case=False, na=False)]
    if "REP" not in df.columns:
        df["REP"] = ""

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
        dropna=False,
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
    group_cols = [
        "CUSTOMER ID",
        "CUSTOMER NAME",
        "LOCATION",
        "LOCATION COORDINATES",
        "REP",
    ]
    aggregations = {
        "TONNAGE": "sum",
        "AMOUNT": "sum",
        "LAT": "first",
        "LONG": "first",
    }
    if "INVOICE NO" in combined_gdf.columns:
        aggregations["INVOICE NO"] = lambda values: ", ".join(
            dict.fromkeys(
                str(value).strip()
                for value in values
                if pd.notna(value) and str(value).strip()
            )
        )
    combined_gdf = combined_gdf.groupby(
        group_cols, as_index=False, dropna=False
    ).agg(aggregations)
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


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _number(value) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(n) else n


def _unique_id_base(now=None) -> int:
    stamp = int(now if now is not None else time.time())
    return stamp if stamp > 0 else 1


def _extract_wialon_error(result):
    if isinstance(result, dict) and result.get("error", 0) != 0:
        return result.get("error", 1), result.get(
            "reason", result.get("message", str(result))
        )
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("error", 0) != 0:
                return item.get("error", 1), item.get(
                    "reason", item.get("message", str(item))
                )
            if isinstance(item, list):
                nested = _extract_wialon_error(item)
                if nested:
                    return nested
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
        route_id = _unique_id_base(current_time)
        order_id_base = route_id
        last_visit_time = int(tf)
        sequence_index = 0

        route_orders.append(
            {
                "id": order_id_base,
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

        for _, cust_row in df_grouped.iterrows():
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
                    "id": order_id_base + sequence_index,
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
        sequence_index += 1
        final_id = order_id_base + sequence_index

        if not end_warehouse_rp:
            end_warehouse_rp = _osrm_polyline(prev_coords, {"y": wh_lat, "x": wh_lon})

        route_orders.append(
            {
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
        route_name = f"{vehicle_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

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
                                "routeId": route_id,
                                "callMode": "create",
                                "exp": 0,
                                "f": 0,
                                "n": route_name,
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

        wialon_error = _extract_wialon_error(route_result)
        if wialon_error:
            error_code, error_message = wialon_error
            return {"error": error_code, "message": error_message}

        created_orders = None
        if isinstance(route_result, dict) and isinstance(route_result.get("orders"), list):
            created_orders = route_result["orders"]
        elif isinstance(route_result, list):
            for item in route_result:
                if isinstance(item, dict) and isinstance(item.get("orders"), list) and item["orders"]:
                    created_orders = item["orders"]
                    break

        if created_orders:
            return {
                "error": 0,
                "message": "Route created successfully",
                "planning_url": planning_url,
                "optimize_result": optimize_result,
                "route_result": route_result,
                "route_uid": route_id,
                "route_name": route_name,
                "ordered_stops": [
                    {
                        "name": _text(row.get("CUSTOMER NAME")),
                        "location": _text(row.get("LOCATION")),
                        "lat": float(row["LAT"]),
                        "lon": float(row["LONG"]),
                        "weightKg": _number(row.get("TONNAGE")) * 1000,
                        "amount": _number(row.get("AMOUNT")),
                        "invoice": _text(row.get("INVOICE NO")),
                    }
                    for _, row in df_grouped.iterrows()
                ],
            }

        return {"error": 1, "message": f"Unexpected or error response: {route_result}"}

    except Exception as e:
        return {"error": 1, "message": f"An unexpected error occurred: {str(e)}"}
