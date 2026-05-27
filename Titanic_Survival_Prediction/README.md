# Titanic Survival Prediction 🚢

This project is a Machine Learning application that predicts whether a passenger would survive the Titanic disaster based on factors like age, gender, passenger class, fare, etc.

I built this project to understand the complete machine learning workflow — from data preprocessing and visualization to model training and deployment using Streamlit.

---

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression model
- Random Forest Classifier
- Model comparison and evaluation
- Confusion Matrix and Classification Report
- Interactive Streamlit web application
- Survival probability prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## Project Structure

Titanic_Survival_Prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── titanic_model.pkl
│
├── src/
│   └── main.py
│
├── app.py
├── requirements.txt
└── README.md

---

## Machine Learning Workflow

1. Loaded and explored the Titanic dataset
2. Performed data preprocessing
3. Handled missing values
4. Converted categorical data into numerical form
5. Trained Logistic Regression and Random Forest models
6. Evaluated models using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
7. Built a Streamlit web application for predictions

---

## Model Performance

### Logistic Regression
- Accuracy: 80%

### Random Forest
- Accuracy: 83%

Random Forest performed better because it handled complex relationships between features more effectively.

---

## How To Run The Project

### 1. Clone the repository

```bash
git clone <repository-link>

### 2. Create virtual environment

python -m venv venv

### 3. Activate virtual environment

venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Run the Streamlit app

streamlit run app.py

## Author

Riddhi Shedame