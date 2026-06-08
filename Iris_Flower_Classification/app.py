import streamlit as st
import pickle

# Load model

with open("models/iris_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌸 Iris Flower Classification App")
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)
st.markdown("""
This Machine Learning app predicts the species of an Iris flower based on flower measurements.
""")

st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.0
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    5.0,
    3.0
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    4.0
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    3.0,
    1.0
)

if st.button("Predict Species"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.success(f"Predicted Species: {prediction[0]}")

    

    st.write(f"Prediction Confidence: {max(probability[0]) * 100:.2f}%")
