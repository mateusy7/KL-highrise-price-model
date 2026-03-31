import streamlit as st
import requests
import json
import os

MAPPING_FILE = 'data/mukim_scheme_mapping.json'
API_URL = "https://6k47viuqqq.ap-southeast-1.awsapprunner.com/predict"

st.set_page_config(page_title="KL Highrise Price Predictor")

if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'r') as f:
        MUKIM_SCHEME_MAP = json.load(f)
else:
    st.error("Mapping file not found!")
    MUKIM_SCHEME_MAP = {}

st.title("KL Highrise Price Prediction")
st.markdown("Enter the property details below to get an estimated market price.")

col_a, col_b = st.columns(2)

with col_a:
    mukim_choice = st.selectbox("Select Mukim", options=list(MUKIM_SCHEME_MAP.keys()))

with col_b:
    raw_list = MUKIM_SCHEME_MAP.get(mukim_choice, [])
    
    clean_list = [item for item in raw_list if item != "Others"]
    
    sorted_names = sorted(clean_list)
    
    available_schemes = sorted_names + ["Others"]
    
    scheme_choice = st.selectbox("Select Area/Scheme", options=available_schemes)

with st.form("remaining_inputs"):
    col1, col2 = st.columns(2)
    
    with col1:
        property_type = st.selectbox("Property Type", [
            "Condominium/Apartment", "Flat", "Low-Cost Flat", "Town House"
        ])
        tenure = st.radio("Tenure", ["Freehold", "Leasehold"])
        
    with col2:
        area = st.number_input("Land Parcel Area (sqft)", min_value=100.0, value=1000.0, max_value=10000.0, step=10.0)
        level = st.slider("Unit Level (Floor)", 0, 50, 15)

    submit = st.form_submit_button("Estimate Price")

if submit:

    payload = {
        "property_type": property_type,
        "mukim": mukim_choice,
        "scheme_name_area": scheme_choice, 
        "tenure": tenure,
        "land_parcel_area": float(area),
        "unit_level": int(level)
    }

    with st.spinner("Calculating price..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                prediction = response.json().get("estimated_price_rm")
                st.success(f"### Fair Price: RM{prediction:,.0f}")
                st.info("Note: This is an estimate based on historical transaction data from 2021-2025. " \
                "The model showed an average error rate of 16% on test data. Best for predictions below RM4,000,000.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")