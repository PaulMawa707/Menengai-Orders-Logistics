import streamlit as st
from app2 import run_wialon_uploader
from test2 import run_tonnage_lookup

st.set_page_config(page_title="Unified Vehicle Tool", layout="wide")
st.title("🚚 Menengai Logistics Dashboard")

app_choice = st.sidebar.radio("Select Tool", ["📦 Wialon PDF Order Uploader", "🚛 Vehicle Tonnage Lookup"])

if app_choice == "📦 Wialon PDF Order Uploader":
    run_wialon_uploader()
elif app_choice == "🚛 Vehicle Tonnage Lookup":
    run_tonnage_lookup()
