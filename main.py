import streamlit as st
from app2 import run_wialon_uploader
from test2 import run_tonnage_lookup
import base64

st.set_page_config(page_title="Unified Vehicle Tool", layout="wide")
st.title("🚚 Menengai Logistics Dashboard")

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

# ✅ Call the function here
set_background()

# Sidebar navigation
app_choice = st.sidebar.radio("Select Tool", ["📦 Logistics PDF Order Uploader", "🚛 Vehicle Tonnage Lookup"])

if app_choice == "📦 Logistics PDF Order Uploader":
    run_wialon_uploader()
elif app_choice == "🚛 Vehicle Tonnage Lookup":
    run_tonnage_lookup()
