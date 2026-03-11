import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import plotly.graph_object as go

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------- BACKGROUND IMAGE FUNCTION ----------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-container {{
        background-color: rgba(255,255,255,0.85);
        padding: 30px;
        border-radius: 12px;
    }}

    .title {{
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
    }}

    .subtitle {{
        text-align: center;
        font-size: 18px;
        color: #444;
        margin-bottom: 30px;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

set_background("house.jpg")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- MAIN CONTAINER ----------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<p class="title">🏠 House Building Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the cost of building a house using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sq ft)", 500, 5000, 1200)
rooms = st.sidebar.slider("Number of Rooms", 1, 10, 3)
floors = st.sidebar.slider("Number of Floors", 1, 5, 1)

material = st.sidebar.selectbox(
    "Material Quality",
    ["Low", "Medium", "High"]
)

labor = st.sidebar.number_input("Labor Cost", 100000, 500000, 200000)

# Convert material
material_value = {"Low":1, "Medium":2, "High":3}[material]

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# INPUT SUMMARY
with col1:
    st.subheader("📋 Input Summary")
    df = pd.DataFrame({
        "Feature": ["Area", "Rooms", "Floors", "Material", "Labor Cost"],
        "Value": [area, rooms, floors, material, labor]
    })
    st.dataframe(df, use_container_width=True)

# FEATURE VISUALIZATION
with col2:
    st.subheader("📊 Feature Visualization")
    chart_data = pd.DataFrame({
        "Area": [area],
        "Rooms": [rooms],
        "Floors": [floors]
    })
    st.bar_chart(chart_data)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("Predict House Price"):
    input_data = np.array([[area, rooms, floors, material_value, labor]])
    prediction = model.predict(input_data)[0]
    price = int(prediction)

    st.success(f"💰 Estimated Building Cost: ₹ {price:,}")

    # ---------- PRICE GAUGE ----------
    # Determine dynamic color
    if price < 1500000:
        gauge_color = "green"
    elif price < 2500000:
        gauge_color = "yellow"
    else:
        gauge_color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        number={'prefix': "₹ "},
        title={'text': "Predicted Cost"},
        gauge={
            'axis': {'range': [0, 4000000]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 1500000], 'color': 'lightgreen'},
                {'range': [1500000, 2500000], 'color': 'lightyellow'},
                {'range': [2500000, 4000000], 'color': 'lightcoral'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# LINE CHART FOR FEATURES
st.subheader("📈 Feature Comparison")
compare_df = pd.DataFrame({
    "Feature": ["Area", "Rooms", "Floors"],
    "Value": [area, rooms, floors]
})
st.line_chart(compare_df.set_index("Feature"))

st.markdown("---")
st.caption("Machine Learning Mini Project | House Building Price Prediction")


st.markdown("</div>", unsafe_allow_html=True)
