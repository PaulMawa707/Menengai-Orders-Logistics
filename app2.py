import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import pdfplumber
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
import os
import base64

# Set timezone environment variable
os.environ['TZ'] = 'Africa/Nairobi'
try:
    time.tzset()  # Unix-specific, will fail on Windows
except AttributeError:
    pass  # Skip on Windows

st.set_page_config(page_title="Wialon Logistics Uploader", layout="wide")

# Convert image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background image
def set_background():
    background_image = get_base64_image("pexels-pixabay-236722.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Show logo at top-right corner
def show_logo_top_right(image_path, width=120):
    logo_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div></div> <!-- empty left spacer -->
            <div style="margin-right: 1rem;">
                <img src="data:image/png;base64,{logo_base64}" width="{width}">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ✅ Apply background and logo
set_background()
show_logo_top_right("CT-Logo.jpg", width=120)  # 

st.markdown("<br>", unsafe_allow_html=True)

def extract_coordinates(coord_str):
    try:
        if "LAT:" in coord_str and "LONG:" in coord_str:
            parts = coord_str.split("LONG:")
            latitude = float(parts[0].replace("LAT:", "").strip().replace(" ", ""))
            longitude = float(parts[1].strip().replace(" ", ""))
            return latitude, longitude
    except:
        pass
    return None, None

def read_excel_to_df(excel_file):
    import re
    raw_df = pd.read_excel(excel_file, header=None)

    # Extract truck number
    truck_number = None
    for row in raw_df[0].astype(str):
        if "Truck Number:" in row:
            truck_number = row.split("Truck Number:")[1].split()[0].strip()
            break

    # Find header row (NO.)
    header_row_idx = raw_df[raw_df.iloc[:,0].astype(str).str.upper().str.strip() == "NO."].index[0]
    df = pd.read_excel(excel_file, header=header_row_idx)

    # ✅ Normalize headers
    df.columns = [
        re.sub(r"\s+", " ", str(col)).replace("\u00A0", " ").strip().upper()
        for col in df.columns
    ]
    df = df.loc[:, ~df.columns.str.startswith("UNNAMED")]  # drop Unnamed cols
    #st.write("Detected columns after cleaning:", df.columns.tolist())  # Debug

    # Drop TOTALS + empty rows
    df = df[df["CUSTOMER ID"].notna()]
    df = df[~df["CUSTOMER NAME"].astype(str).str.contains("TOTAL", case=False, na=False)]

    # Clean numeric
    df['TONNAGE'] = pd.to_numeric(df['TONNAGE'].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)

    # Group
    df_grouped = df.groupby(
        ['CUSTOMER ID', 'CUSTOMER NAME', 'LOCATION', 'LOCATION COORDINATES', 'REP'],
        as_index=False
    ).agg({
        'TONNAGE': 'sum',
        'AMOUNT': 'sum',
        'INVOICE NO': lambda x: ', '.join(str(i) for i in x if pd.notna(i))
    })

    # Extract coords
    df_grouped[['LAT', 'LONG']] = df_grouped['LOCATION COORDINATES'].apply(
        lambda x: pd.Series(extract_coordinates(x))
    )
    df_grouped = df_grouped.dropna(subset=['LAT', 'LONG'])

    return df_grouped, truck_number


def read_asset_id_from_excel(excel_file, truck_number):
    df = pd.read_excel(excel_file)
    df.columns = [col.strip().lower() for col in df.columns]

    if "reportname" not in df.columns or "itemid" not in df.columns:
        raise ValueError("Excel file must contain 'ReportName' and 'itemId' columns.")

    match = df[df["reportname"].str.upper().str.contains(truck_number.upper(), na=False)]
    if not match.empty:
        return int(match.iloc[0]["itemid"]), match.iloc[0]["reportname"]
    return None, None

def process_multiple_excels(excel_files):
    all_gdfs = []
    truck_numbers = set()
    for excel_file in excel_files:
        gdf_joined, truck_number = read_excel_to_df(excel_file)
        if gdf_joined is not None:
            all_gdfs.append(gdf_joined)
            if truck_number:
                truck_numbers.add(truck_number)
    if not all_gdfs:
        raise ValueError("No valid data found in any of the Excel files.")
    if len(truck_numbers) > 1:
        raise ValueError(f"Multiple truck numbers found: {', '.join([str(t) for t in truck_numbers])}")
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    combined_gdf = combined_gdf.drop_duplicates(subset=['CUSTOMER ID', 'LOCATION'], keep='first')
    return combined_gdf, list(truck_numbers)[0] if truck_numbers else None

def send_orders_and_create_route(token, resource_id, unit_id, vehicle_name, df_grouped, tf, tt):
    try:
        # Base URL for Wialon API
        base_url = "https://hst-api.wialon.com/wialon/ajax.html"

        # First login with token to get session
        login_payload = {
            "svc": "token/login",
            "params": json.dumps({
                "token": str(token).strip()
            })
        }
        
        st.info("Logging in with token...")
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        login_response = requests.post(base_url, data=login_payload, headers=headers)
        login_result = login_response.json()
        session_id = login_result['eid']

        

        # Define MORL warehouse location
        morl_lat = -0.28802969095623043
        morl_lon = 36.04494759379902

        # Calculate distances from MORL to each delivery point
        def calculate_distance(lat1, lon1, lat2, lon2):
            from math import sin, cos, sqrt, atan2, radians
            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c

            return distance

        # Add distance from MORL to each point
        df_grouped['Distance_From_MORL'] = df_grouped.apply(
            lambda row: calculate_distance(morl_lat, morl_lon, row['LAT'], row['LONG']),
            axis=1
        )

        # Sort by distance from MORL
        df_grouped = df_grouped.sort_values('Distance_From_MORL').reset_index(drop=True)

        # Step 1: Create orders for optimization (customers only; MORL handled as warehouse)
        orders = []
        morl_coords = f"{morl_lat}, {morl_lon}"

        # Add delivery points with sequential IDs
        priority_dict = {str(unit_id): {}}
        order_id_to_name = {}

        for idx, row in df_grouped.iterrows():
            try:
                weight_kg = int(float(row.get('TONNAGE', 0)) * 1000)
            except:
                weight_kg = 0

            # Format coordinates for address field
            coords = f"{row['LAT']}, {row['LONG']}"
            location = f"{row['LOCATION']} ({coords})"

            order_id = idx + 1
            orders.append({
                "y": float(row['LAT']),
                "x": float(row['LONG']),
                "tf": tf,
                "tt": tt,
                "n": row['CUSTOMER NAME'],
                "f": 0,
                "r": 20,
                "id": order_id,
                "p": {
                    "ut": 900,
                    "rep": True,
                    "w": weight_kg,
                    "v": 0,
                    "pr": idx + 1,
                    "criterions": {
                        "max_late": 0,
                        "use_unloading_late": 0
                    },
                    "a": location  # Add coordinates to address
                },
                "cmp": {"unitRequirements": {"values": []}}
            })
            priority_dict[str(unit_id)][str(order_id)] = idx + 1
            order_id_to_name[order_id] = row['CUSTOMER NAME']
        # MORL is not added as order; it will be provided via warehouses and as start/end in route creation

        # Step 2: Optimize route
        optimize_payload = {
            "svc": "order/optimize",
            "params": json.dumps({
                "itemId": int(resource_id),
                "orders": orders,
                "warehouses": [
                    {
                        "id": 0,
                        "y": morl_lat,
                        "x": morl_lon,
                        "n": "MORL",
                        "f": 260,
                        "a": f"MORL ({morl_coords})"
                    },
                    {
                        "id": 99999,
                        "y": morl_lat,
                        "x": morl_lon,
                        "n": "MORL",
                        "f": 264,
                        "a": f"MORL ({morl_coords})"
                    }
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
                    "traffic_model": "best_guess"
                },
                "priority": {},
                "criterions": {
                    "penalties_profile": "balanced"
                },
                "pf": {
                    "n": "MORL",
                    "y": morl_lat,
                    "x": morl_lon,
                    "a": f"MORL ({morl_coords})"  # Add coordinates to address
                },
                "pt": {
                    "n": "MORL",
                    "y": morl_lat,
                    "x": morl_lon,
                    "a": f"MORL ({morl_coords})"  # Add coordinates to address
                },
                "tf": tf,
                "tt": tt
            }),
            "sid": session_id
        }

        st.info("Optimizing route...")
        # (debug outputs removed)
        optimize_response = requests.post(base_url, data=optimize_payload)
        optimize_result = optimize_response.json()

        # Check for optimization errors
        if 'error' in optimize_result:
            if optimize_result['error'] != 0:
                error_msg = optimize_result.get('reason', 'Unknown error during optimization')
                st.error(f"Optimization failed: {error_msg}")
                if 'details' in optimize_result:
                    st.write("Error details:", optimize_result['details'])
                return {"error": optimize_result['error'], "message": error_msg}

        # Extract optimized orders and route summary
        optimized_orders = []
        route_summary = None
        end_warehouse_rp = None

        # The response structure shows orders directly in the unit_id object
        if isinstance(optimize_result, dict):
            unit_key = str(unit_id)
            if unit_key in optimize_result and 'orders' in optimize_result[unit_key]:
                optimized_orders = optimize_result[unit_key]['orders']
                if 'routes' in optimize_result[unit_key]:
                    route_summary = optimize_result[unit_key]['routes'][0]
                # capture final warehouse polyline if present
                def _extract_rp(o):
                    if not isinstance(o, dict):
                        return None
                    return o.get('rp') or o.get('p')
                for resp_order in reversed(optimized_orders):
                    if isinstance(resp_order, dict) and resp_order.get('f') == 264:
                        rp_val = _extract_rp(resp_order)
                        if rp_val:
                            end_warehouse_rp = rp_val
                            break
                        break

        if not optimized_orders:
            st.error("No optimized orders found in response")
            return {"error": 1, "message": "No optimized orders found"}

        # (debug outputs removed)

        # Create a mapping of coordinates from the original orders
        coord_map = {}
        for idx, row in df_grouped.iterrows():
            coord_map[row['CUSTOMER NAME']] = {
                'y': float(row['LAT']),
                'x': float(row['LONG'])
            }

        # Add warehouse coordinates
        coord_map['MORL'] = {'y': morl_lat, 'x': morl_lon}
        

        # Step 3: Create final route with routing information
        route_orders = []
        current_time = int(time.time())
        route_id = current_time

        # Start with MORL as initial warehouse order in route
        route_orders = []
        current_time = int(time.time())
        route_id = current_time

        # Add initial warehouse (MORL) as first order with f:260
        last_visit_time = int(tf)
        sequence_index = 0
        route_orders.append({
            "uid": int(unit_id),
            "id": 0,
            "n": "MORL",
            "p": {
                "ut": 0,
                "rep": True,
                "w": "0",
                "c": "0",
                "r": {
                    "vt": last_visit_time,
                    "ndt": 60,
                    "id": route_id,
                    "i": sequence_index,
                    "m": 0,
                    "t": 0
                },
                "u": int(unit_id),
                "a": f"MORL ({morl_lat}, {morl_lon})",
                "weight": "0",
                "cost": "0"
            },
            "f": 260,
            "tf": tf,
            "tt": tt,
            "r": 100,
            "y": morl_lat,
            "x": morl_lon,
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
            "cargo": {"weight": "0", "cost": "0"}
        })

        prev_coords = { 'y': morl_lat, 'x': morl_lon }

        # Process optimized customer orders only (skip warehouses returned by optimizer)
        for idx, order in enumerate(optimized_orders):
            if isinstance(order, dict):
                order_id = order.get('id')
            else:
                order_id = None
            # Skip if this id is not a customer order we sent (e.g., warehouse id 0 or synthetic ids)
            if order_id is None or order_id not in order_id_to_name:
                continue
            order_name = order_id_to_name[order_id]
            coords = coord_map.get(order_name, {'y': morl_lat, 'x': morl_lon})
            
            # Get weight and cost from original orders (customers only)
            weight_kg = 0
            cost = 0
            try:
                customer_data = df_grouped[df_grouped['CUSTOMER NAME'] == order_name]
                if not customer_data.empty:
                    weight_kg = int(float(customer_data.iloc[0]['TONNAGE']) * 1000)
                    cost = float(customer_data.iloc[0]['AMOUNT'])
            except Exception:
                weight_kg = 0
                cost = 0

            # Format coordinates for address
            location = f"{order_name} ({coords['y']}, {coords['x']})"
            try:
                location_data = df_grouped[df_grouped['CUSTOMER NAME'] == order_name]
                if not location_data.empty:
                    location = f"{location_data.iloc[0]['LOCATION']} ({coords['y']}, {coords['x']})"
            except Exception:
                pass
            
            if isinstance(order, dict):
                order_tm = order.get('tm')
                order_ml = order.get('ml', 0)
                order_rp = order.get('rp') or order.get('p')
            else:
                order_tm = None
                order_ml = 0
                order_rp = None

            # Normalize planned visit time so it's never zero
            if not isinstance(order_tm, int) or order_tm <= 0:
                order_tm = last_visit_time + 600
            else:
                # ensure non-decreasing and after start
                order_tm = max(order_tm, last_visit_time + 60, int(tf))

            # Calculate distance from previous point
            distance = calculate_distance(
                prev_coords['y'], prev_coords['x'],
                coords['y'], coords['x']
            )
            mileage = int(distance * 1000)

            # Fallback: if optimizer did not provide rp for this leg, fetch from OSRM
            if not order_rp:
                try:
                    osrm_url_leg = (
                        f"https://router.project-osrm.org/route/v1/driving/"
                        f"{prev_coords['x']},{prev_coords['y']};{coords['x']},{coords['y']}?overview=full&geometries=polyline"
                    )
                    osrm_resp_leg = requests.get(osrm_url_leg, timeout=15)
                    osrm_json_leg = osrm_resp_leg.json()
                    if isinstance(osrm_json_leg, dict) and osrm_json_leg.get('routes'):
                        order_rp = osrm_json_leg['routes'][0].get('geometry')
                        st.info(f"Using OSRM fallback polyline for leg to order {order_id}.")
                except Exception as _osrm_leg_err:
                    st.write("Could not fetch fallback polyline for leg:", str(_osrm_leg_err))
            
            sequence_index += 1
            route_order = {
                "uid": int(unit_id),
                "id": order_id,
                "n": order_name,
                "p": {
                    "ut": 900,
                    "rep": True,
                    "w": str(weight_kg),
                    "c": str(int(cost)),
                    "r": {
                        "vt": order_tm,
                        "ndt": 60,
                        "id": route_id,
                        "i": sequence_index,
                        "m": mileage,
                        "t": 0
                    },
                    "u": int(unit_id),
                    "a": location,
                    "weight": str(weight_kg),
                    "cost": str(int(cost))
                },
                "f": 0,
                "tf": tf,
                "tt": tt,
                "r": 20,
                "y": coords['y'],
                "x": coords['x'],
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
                "cost": str(int(cost)),
                "cargo": {
                    "weight": str(weight_kg),
                    "cost": str(int(cost))
                }
            }
            route_orders.append(route_order)
            prev_coords = coords
            last_visit_time = order_tm

        # Add final warehouse (MORL) as last order with f:264
        distance_back = calculate_distance(prev_coords['y'], prev_coords['x'], morl_lat, morl_lon)
        mileage_back = int(distance_back * 1000)
        final_id = max([o["id"] if isinstance(o, dict) and "id" in o else 0 for o in route_orders] + [0]) + 1
        sequence_index += 1

        # If optimizer didn't provide a polyline for the final leg, try to fetch one from OSRM as a fallback
        if not end_warehouse_rp:
            try:
                osrm_url = (
                    f"https://router.project-osrm.org/route/v1/driving/"
                    f"{prev_coords['x']},{prev_coords['y']};{morl_lon},{morl_lat}?overview=full&geometries=polyline"
                )
                osrm_resp = requests.get(osrm_url, timeout=15)
                osrm_json = osrm_resp.json()
                if isinstance(osrm_json, dict) and osrm_json.get('routes'):
                    end_warehouse_rp = osrm_json['routes'][0].get('geometry')
                    st.info("Using OSRM fallback polyline for final leg to warehouse.")
            except Exception as _osrm_err:
                st.write("Could not fetch fallback polyline:", str(_osrm_err))
        route_orders.append({
            "uid": int(unit_id),
            "id": final_id,
            "n": "MORL",
            "p": {
                "ut": 0,
                "rep": True,
                "w": "0",
                "c": "0",
                "r": {
                    "vt": last_visit_time + 600,
                    "ndt": 60,
                    "id": route_id,
                    "i": sequence_index,
                    "m": mileage_back,
                    "t": 0
                },
                "u": int(unit_id),
                "a": f"MORL ({morl_lat}, {morl_lon})",
                "weight": "0",
                "cost": "0"
            },
            "f": 264,
            "tf": tf,
            "tt": tt,
            "r": 100,
            "y": morl_lat,
            "x": morl_lon,
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
            "cargo": {"weight": "0", "cost": "0"}
        })

        # (debug outputs removed)

        # Calculate total mileage and cost
        total_mileage = sum(order['p']['r']['m'] for order in route_orders)
        total_cost = sum(float(order['p']['c']) for order in route_orders if order['f'] == 0)
        total_weight = sum(int(order['p']['w']) for order in route_orders if order['f'] == 0)

        # Create the final route
        batch_payload = {
            "svc": "core/batch",
            "params": json.dumps({
                "params": [{
                    "svc": "order/route_update",
                    "params": {
                        "itemId": int(resource_id),
                        "orders": route_orders,
                        "uid": route_id,
                        "callMode": "create",
                        "exp": 3600,
                        "f": 0,
                        "n": f"{vehicle_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "summary": {
                            "countOrders": len(route_orders),
                            "duration": route_summary['duration'] if route_summary and 'duration' in route_summary else 0,
                            "mileage": total_mileage,
                            "priceMileage": float(total_mileage) / 1000,
                            "priceTotal": total_cost,
                            "weight": total_weight,
                            "cost": total_cost
                        }
                    }
                }],
                "flags": 0
            }),
            "sid": session_id
        }

        st.info("Creating final route...")
        route_response = requests.post(base_url, data=batch_payload)
        route_result = route_response.json()
        # (debug outputs removed)

        if isinstance(route_result, list):
            has_error = any(isinstance(item, dict) and item.get('error', 0) != 0 for item in route_result)
            if not has_error:
                planning_url = (
                    f"https://apps.wialon.com/logistics/?"
                    f"lang=en&"
                    f"sid={session_id}#"
                    f"/distrib/step3"
                )
                return {
                    "error": 0,
                    "message": "Route created successfully",
                    "planning_url": planning_url,
                    "optimize_result": optimize_result,
                    "route_result": route_result
                }
            else:
                error_item = next((item for item in route_result if isinstance(item, dict) and item.get('error', 0) != 0), None)
                return {
                    "error": error_item.get('error', 1) if error_item else 1,
                    "message": error_item.get('reason', 'Unknown error') if error_item else 'Unknown error in batch response'
                }
        elif isinstance(route_result, dict):
            if route_result.get("error", 1) == 0:
                planning_url = (
                    f"https://apps.wialon.com/logistics/?"
                    f"lang=en&"
                    f"sid={session_id}#"
                    f"/distrib/step3"
                )
                return {
                    "error": 0,
                    "message": "Route created successfully",
                    "planning_url": planning_url,
                    "optimize_result": optimize_result,
                    "route_result": route_result
                }
            else:
                return route_result
        else:
            return {
                "error": 1,
                "message": f"Unexpected response type: {type(route_result)}"
            }

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Error details:", str(e))
        st.write("Error location:", e.__traceback__.tb_lineno)
        return {"error": 1, "message": f"An unexpected error occurred: {str(e)}"}

def process_multiple_excels(excel_files):
    all_gdfs = []
    truck_numbers = set()
    for excel_file in excel_files:
        gdf_joined, truck_number = read_excel_to_df(excel_file)
        if gdf_joined is not None:
            all_gdfs.append(gdf_joined)
            if truck_number:
                truck_numbers.add(truck_number)
    if not all_gdfs:
        raise ValueError("No valid data found in any of the Excel files.")
    if len(truck_numbers) > 1:
        raise ValueError(f"Multiple truck numbers found: {', '.join([str(t) for t in truck_numbers])}")
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    combined_gdf = combined_gdf.drop_duplicates(subset=['CUSTOMER ID', 'LOCATION'], keep='first')
    return combined_gdf, list(truck_numbers)[0] if truck_numbers else None

def run_wialon_uploader():
    st.subheader("\U0001F4E6 Logistics Excel Orders Uploader (via Logistics API)")
    with st.form("upload_form"):
        excel_files = st.file_uploader("Upload Excel File(s) - All must be for the same truck", type=["xls", "xlsx"], accept_multiple_files=True)
        assets_file = st.file_uploader("Upload Excel File (Assets)", type=["xls", "xlsx"])
        selected_date = st.date_input("Select Route Date")
        col1, col2 = st.columns(2)
        with col1: start_hour = st.slider("Route Start Hour", 0, 23, 6)
        with col2: end_hour = st.slider("Route End Hour", start_hour + 1, 23, 18)
        token = st.text_input("Enter your Wialon Token", type="password")
        resource_id = st.text_input("Enter Wialon Resource ID")
        submit_btn = st.form_submit_button("Upload and Dispatch")

    if submit_btn:
        if not excel_files or not assets_file or not token or not resource_id:
            st.error("Please upload orders Excel, assets Excel, token, and resource ID.")
        else:
            try:
                with st.spinner("Processing..."):
                    tz = pytz.timezone('Africa/Nairobi')
                    start_time = tz.localize(datetime.combine(selected_date, datetime.min.time().replace(hour=start_hour)))
                    end_time = tz.localize(datetime.combine(selected_date, datetime.min.time().replace(hour=end_hour)))
                    tf, tt = int(start_time.timestamp()), int(end_time.timestamp())
                    gdf_joined, truck_number = process_multiple_excels(excel_files)
                    if gdf_joined is None: return
                    unit_id, vehicle_name = read_asset_id_from_excel(assets_file, truck_number)
                    if not unit_id:
                        st.error(f"Could not find unit ID for truck: {truck_number}")
                        return
                    st.info("Summary of orders:")
                    st.write(f"Delivery points: {len(gdf_joined)}")
                    st.write(f"Tonnage: {gdf_joined['TONNAGE'].sum():.2f}")
                    st.write(f"Amount: {gdf_joined['AMOUNT'].sum():.2f}")
                    result = send_orders_and_create_route(token, int(resource_id), unit_id, vehicle_name, gdf_joined, tf, tt)
                    if result.get("error") == 0:
                        st.success("✅ Route created successfully!")
                        st.markdown(f"[Open Wialon Logistics]({result['planning_url']})", unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_wialon_uploader()


