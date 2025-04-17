import streamlit as st
import pandas as pd
import pdfplumber
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import io

# === Function: Read PDF and clean ===
def read_pdf_to_df(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[0]
        tables = page.extract_tables()
        if not tables:
            raise ValueError("No tables found in the PDF")

        raw_table = tables[0]
        headers = raw_table[0]
        data_rows = raw_table[1:]
        df2 = pd.DataFrame(data_rows, columns=headers)

    df_raw = df2
    df_raw.columns = df_raw.iloc[0]
    df_cleaned = df_raw.drop([0, 1])
    df_cleaned.reset_index(drop=True, inplace=True)
    df_cleaned.columns = df_cleaned.columns.str.replace('\n', ' ').str.strip()
    df_cleaned.columns = [col.replace('\n', ' ').replace('REMA RKS', 'REMARKS').strip() if isinstance(col, str) else col for col in df_cleaned.columns]
    df_cleaned = df_cleaned.applymap(lambda x: x.replace('\n', ' ').strip() if isinstance(x, str) else x)

    return df_cleaned

# === Function: Extract coordinates from string ===
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

# === Function: Convert cleaned DataFrame to Wialon orders ===
def convert_to_orders(df_cleaned, tf, tt):
    orders = []
    for idx, row in df_cleaned.iterrows():
        latitude, longitude = extract_coordinates(row.get('COORDINATES', ''))
        if latitude is None or longitude is None:
            st.warning(f"Skipping row {idx} due to invalid coordinates.")
            continue

        order = {
            "itemId": 25601229,
            "id": 0,
            "n": row['CUSTOMER NAME'],
            "oldOrderId": 0,
            "oldOrderFiles": [],
            "p": {
                "n": row['CUSTOMER NAME'],
                "p": "",
                "p2": "",
                "e": "",
                "a": f"{row['LOCATION']} (LAT: {latitude}, LONG: {longitude})",
                "v": 50,
                "w": row['TONNAGE'],
                "c": row['AMOUNT'],
                "ut": 1800,
                "t": "",
                "d": row['DELIVERY'] if row['DELIVERY'] else "No description",
                "uic": str(row['INVOICE NO.']),
                "cid": str(row['CUSTOMER ID']),
                "cm": "Handle with care",
                "aff": "123,456",
                "z": "1001_2002",
                "ntf": 3,
                "pr": 1
            },
            "rp": "",
            "f": 161,
            "tf": tf,
            "tt": tt,
            "trt": 600,
            "r": 100,
            "y": latitude,
            "x": longitude,
            "ej": {},
            "tz": 3,
            "cf": {
                "delivery_notes": "",
                "payment_status": ""
            },
            "callMode": "create",
            "dp": []
        }
        orders.append(order)
    return orders

# === Function: Login to Wialon API ===
def login_to_wialon(token):
    login_url = "https://hst-api.wialon.com/wialon/ajax.html"
    params = {
        "svc": "token/login",
        "params": json.dumps({"token": token})
    }
    response = requests.get(login_url, params=params)
    data = response.json()
    if 'eid' not in data:
        raise Exception(f"Login failed: {data}")
    return data['eid']

# === Function: Send orders to Wialon ===
def send_orders(orders, sid):
    order_url = f"https://hst-api.wialon.com/wialon/ajax.html?sid={sid}"
    for order in orders:
        payload = {
            "svc": "order/update",
            "params": json.dumps(order)
        }
        response = requests.get(order_url, params=payload)
        result = response.json()
        st.success(f"Sent order: {order['n']}, Response: {result}")
        time.sleep(1)

# === Streamlit UI ===
st.set_page_config(page_title="Wialon Order Uploader", layout="wide")
st.title("📦 Wialon PDF Order Uploader")

with st.form("upload_form"):
    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])
    selected_date = st.date_input("Select Date (Kenya Time)")
    token = st.text_input("Enter your Wialon token", type="password")
    submit_btn = st.form_submit_button("Upload and Send Orders")

if submit_btn:
    if not pdf_file or not token:
        st.error("Please upload a PDF and enter a token.")
    else:
        try:
            # Convert to Unix timestamps for the date (Kenya timezone)
            nairobi_tz = pytz.timezone("Africa/Nairobi")
            start_dt = nairobi_tz.localize(datetime.combine(selected_date, datetime.min.time()))
            end_dt = start_dt + timedelta(days=1)

            tf = int(start_dt.timestamp())
            tt = int(end_dt.timestamp())

            # Read and process PDF
            st.info("Processing PDF...")
            df_cleaned = read_pdf_to_df(pdf_file)
            st.success("PDF parsed successfully!")

            # Convert to Wialon orders
            orders = convert_to_orders(df_cleaned, tf, tt)
            st.info(f"{len(orders)} valid orders ready.")

            # Login and send orders
            st.info("Logging in to Wialon...")
            sid = login_to_wialon(token)
            st.success("Logged in!")

            st.info("Sending orders to Wialon...")
            send_orders(orders, sid)
            st.success("All done!")

        except Exception as e:
            st.error(f"Error: {e}")
