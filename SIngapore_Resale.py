import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
# st.set_page_config(layout="wide")

# Load Model
@st.cache_data
def load_model():
    with open("Resale_Flat_Prices_Model.pkl", "rb") as f:
        return pickle.load(f)

regg_model = load_model()

# Mappings
TOWN_MAPPING = {
    'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
    'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
    'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
    'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17,
    'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21,
    'TAMPINES': 22, 'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
}

FLAT_TYPE_MAPPING = {
    '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4,
    'EXECUTIVE': 5, 'MULTI-GENERATION': 6
}

FLAT_MODEL_MAPPING = {
    'Improved': 5, 'New Generation': 12, 'Model A': 8, 'Standard': 17, 'Simplified': 16,
    'Premium Apartment': 13, 'Maisonette': 7, 'Apartment': 3, 'Model A2': 10,
    'Type S1': 19, 'Type S2': 20, 'Adjoined flat': 2, 'Terrace': 18, 'DBSS': 4,
    'Model A-Maisonette': 9, 'Premium Maisonette': 15, 'Multi Generation': 11,
    'Premium Apartment Loft': 14, 'Improved-Maisonette': 6, '2-room': 0, '3Gen': 1
}

# Prediction Function
def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year, re_les_month, les_coms_dt):
    user_data = np.array([[
        int(year), TOWN_MAPPING[town], FLAT_TYPE_MAPPING[flat_type], float(flr_area_sqm),
        FLAT_MODEL_MAPPING[flat_model], np.log(max(float(stry_start), 1e-6)),
        np.log(max(float(stry_end), 1e-6)), int(re_les_year), int(re_les_month), int(les_coms_dt)
    ]], dtype=np.float32)
    
    price = np.exp(regg_model.predict(user_data)[0])
    return round(price)

# Streamlit UI
# st.set_page_config(layout="wide")
st.markdown("# :house: Singapore Resale Flat Price Prediction")
st.write("---")

with st.sidebar:
    select = option_menu(
        "MAIN MENU", ["Home", "Price Prediction", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="menu-button-wide", default_index=0
    )

if select == "Home":
    img= Image.open(r"C:\Sudhakar\Projects\Singapore  Resale Flat Prices Predicting\Dataset & Documents\Singapore_image.jpg")
    st.image(img, width=600)

    st.markdown("<h2 style='color:green;'>HDB Flats:</h2>", unsafe_allow_html=True)

    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')
    
    st.markdown("<h2 style='color:green;'>Resale Process:</h2>", unsafe_allow_html=True)

    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')
    
    st.markdown("<h2 style='color:green;'>Valuation:</h2>", unsafe_allow_html=True)

    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')
    
    st.markdown("<h2 style='color:green;'>Eligibility Criteria:</h2>", unsafe_allow_html=True)

    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")
    
    st.markdown("<h2 style='color:green;'>Resale Levy:</h2>", unsafe_allow_html=True)

    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")
    
    st.markdown("<h2 style='color:green;'>Grant Schemes:</h2>", unsafe_allow_html=True)

    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")
    
    st.markdown("<h2 style='color:green;'>HDB Loan and Bank Loan:</h2>", unsafe_allow_html=True)

    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")
    
    st.markdown("<h2 style='color:green;'>Market Trends:</h2>", unsafe_allow_html=True)

    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")
    
    st.markdown("<h2 style='color:green;'>Online Platforms:</h2>", unsafe_allow_html=True)

    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")

elif select == "Price Prediction":
    st.markdown("## 📊 Predict Resale Flat Price")
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.selectbox("Select the Year", list(map(str, range(2015, 2026))))
        town = st.selectbox("Select the Town", list(TOWN_MAPPING.keys()))
        flat_type = st.selectbox("Select the Flat Type", list(FLAT_TYPE_MAPPING.keys()))
        flr_area_sqm = st.number_input("Floor Area (sqm)", min_value=31, max_value=280, value=50)
        flat_model = st.selectbox("Select the Flat Model", list(FLAT_MODEL_MAPPING.keys()))
    
    with col2:
        stry_start = st.number_input("Storey Start", min_value=1, max_value=50, value=5)
        stry_end = st.number_input("Storey End", min_value=1, max_value=50, value=10)
        re_les_year = st.number_input("Remaining Lease Year", min_value=42, max_value=97, value=50)
        re_les_month = st.number_input("Remaining Lease Month", min_value=0, max_value=11, value=6)
        les_coms_dt = st.selectbox("Lease Commence Date", list(map(str, range(1966, 2023))))
    
    if st.button("Predict Price", use_container_width=True):
        predicted_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model,
                                       stry_start, stry_end, re_les_year, re_les_month, les_coms_dt)
        st.success(f"### 🏷️ Predicted Price: {predicted_price:,}")


elif select == "About":

    st.header(":blue[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")

    st.header(":blue[Feature Engineering:]")
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    
    st.header(":blue[Model Selection and Training:]")
    st.write("Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.")

    st.header(":blue[Model Evaluation:]")
    st.write("Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")

    st.header(":blue[Streamlit Web Application:]")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")

    st.header(":blue[Deployment on Render:]")
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")
    
    st.header(":blue[Testing and Validation:]")
    st.write("Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.")

