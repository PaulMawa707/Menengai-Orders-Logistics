import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import io
import pdfplumber

def read_pdf_to_df(pdf_file):

    def clean_column_names(columns):
        cleaned = []
        for col in columns:
            col = str(col).replace('\n', ' ').strip() if col else ""
            cleaned.append(col)
        return cleaned

    required_columns = [
        "REP",
        "CUSTOMER ID",
        "CUSTOMER NAME",
        "LOCATION",
        "COORDINATES",
        "INVOICE NO.",
        "AMOUNT",
        "TONNAGE"
    ]

    all_rows = []
    header = None

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if not table:
                    continue

                # Find the header
                if i == 0 and not header:
                    for row in table:
                        if row and row[0] and not row[0].startswith("Sales Order Booking Delivery Sheet"):
                            header = clean_column_names(row)
                            break
                    data = table[1:] if header else []
                else:
                    data = table

                # Remove blank rows
                data = [row for row in data if any(cell and str(cell).strip() for cell in row)]
                all_rows.extend(data)

    if not header or not all_rows:
        raise ValueError("Could not extract table properly.")

    # Create initial DataFrame
    df_cleaned = pd.DataFrame(all_rows, columns=clean_column_names(header))

    # Remove repeated header rows and irrelevant rows
    header_keywords = ['REP', 'CUSTOMER ID', 'CUSTOMER NAME', 'INVOICE NO.', 'AMOUNT', 'TONNAGE']
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: any(str(cell).strip().upper() in header_keywords for cell in row), axis=1)]
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.astype(str).str.contains('Fixed|Driver Sign|Mileage|Cartons', case=False).any(), axis=1)]

    # Keep only required columns
    df_cleaned = df_cleaned[[col for col in df_cleaned.columns if col.strip().upper() in required_columns]]

    # Rename columns to uppercase
    df_cleaned.columns = [col.strip().upper() for col in df_cleaned.columns]

    # Clean multiline cells
    for col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()

        # Remove empty rows and reset index
        
        df_cleaned = df_cleaned.dropna(how='all')
        df_cleaned = df_cleaned[df_cleaned['CUSTOMER ID'].notna()]

        # Reset the index
        df_cleaned = df_cleaned.reset_index(drop=True)

        # Final preview
        print("🧪 Cleaned Dataframe Preview (after final cleaning):")
        print(df_cleaned.head(20))


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
            st.error(f"Done: {e}")