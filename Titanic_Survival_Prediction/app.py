import streamlit as st
import pickle

# Load saved model
with open("models/titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Titanic Survival Predictor")
st.sidebar.header("Passenger Information")
st.markdown("""
This Machine Learning app predicts whether a passenger would survive the Titanic disaster using a Random Forest Classifier.
""")


st.write("Enter passenger details below:")

# User Inputs
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

age = st.sidebar.slider("Age", 1, 80, 25)

sibsp = st.sidebar.number_input("Number of Siblings/Spouses", min_value=0, max_value=10, value=0)

parch = st.sidebar.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)

fare =st.sidebar.slider("Fare", 0, 600, 50)

embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Convert categorical inputs

sex_value = 0 if sex == "Male" else 1

embarked_mapping = {
    "S": 0,
    "C": 1,
    "Q": 2
}

embarked_value = embarked_mapping[embarked]

# Prediction Button

if st.button("Predict Survival"):

    input_data = [[
        pclass,
        sex_value,
        age,
        sibsp,
        parch,
        fare,
        embarked_value
    ]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
         st.success("Passenger Survived")
         st.write(f"Survival Probability: {probability[0][1] * 100:.2f}%")
    else:
         st.error("Passenger Did Not Survive")
         st.write(f"Survival Probability: {probability[0][1] * 100:.2f}%")