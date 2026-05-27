# Iris Flower Classification 🌸

This project is a Machine Learning application that predicts the species of an Iris flower based on flower measurements such as sepal length, sepal width, petal length, and petal width.

I built this project to understand multiclass classification, data visualization, and the complete machine learning workflow using Scikit-learn and Streamlit.

---

## Features

- Data loading using Scikit-learn
- Exploratory Data Analysis (EDA)
- Data visualization using Seaborn
- Pairplot visualization
- Multiclass classification
- Random Forest Classifier
- Model evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Interactive Streamlit web application
- Prediction confidence display

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

Iris_Flower_Classification/
│
├── models/
│   └── iris_model.pkl
│
├── src/
│   └── main.py
│
├── app.py
├── requirements.txt
└── README.md

---

## Machine Learning Workflow

1. Loaded the Iris dataset
2. Explored and visualized the dataset
3. Created pairplots to understand feature relationships
4. Split data into training and testing sets
5. Trained a Random Forest Classifier
6. Evaluated the model using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
7. Saved the trained model using Pickle
8. Built a Streamlit web application for predictions

---

## About The Dataset

The Iris dataset contains three flower species:

- Setosa
- Versicolor
- Virginica

The model predicts the flower species using:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Model Performance

- Model Accuracy: 100%

The model achieved very high accuracy because the Iris dataset is clean and the flower species are highly separable.

---

## How To Run The Project

### 1. Clone the repository


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