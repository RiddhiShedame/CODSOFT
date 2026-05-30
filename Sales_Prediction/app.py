import streamlit as st
import pickle

# Page Configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Model
with open("models/sales_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Sales Prediction Dashboard")
st.markdown(
    "Predict sales based on advertising expenditure and gain business insights."
)

st.divider()

# Input Section
st.subheader("Advertising Budget Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input(
        "📺 TV Budget",
        min_value=0.0,
        value=100.0
    )

with col2:
    radio = st.number_input(
        "📻 Radio Budget",
        min_value=0.0,
        value=25.0
    )

with col3:
    newspaper = st.number_input(
        "📰 Newspaper Budget",
        min_value=0.0,
        value=15.0
    )

st.divider()

# Prediction
if st.button("Predict Sales"):

    prediction = model.predict([[tv, radio, newspaper]])

    st.success("Prediction Generated Successfully!")

    st.subheader("Prediction Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TV Budget", f"{tv}")
    c2.metric("Radio Budget", f"{radio}")
    c3.metric("Newspaper Budget", f"{newspaper}")
    c4.metric("Predicted Sales", f"{prediction[0]:.2f}")

    st.divider()

    st.subheader("📈 Business Insight")

    st.info(
        """
        Based on the trained regression model,
        TV advertising has the strongest impact on sales,
        followed by Radio advertising.
        Newspaper advertising contributes comparatively less.
        """
    )

    if tv > radio and tv > newspaper:
        st.info("TV advertising receives the highest budget allocation.")
    elif radio > tv and radio > newspaper:
        st.info("Radio advertising receives the highest budget allocation.")
    else:
        st.info("Newspaper advertising receives the highest budget allocation.")

    st.subheader("Sales Performance Indicator")
    st.progress(min(int(prediction[0] * 3), 100))