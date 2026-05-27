from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df["species"] = iris.target

# Show first 5 rows
print(df.head())

print("\nDataset Info:\n")

print(df.info())

print("\nMissing Values:\n")

print(df.isnull().sum())

species_mapping = {0 : "Setosa", 1 : "Versicolor", 2 : "Virginica"}

# EDA....

df["species"] = df["species"].map(species_mapping)

sns.pairplot(df, hue="species")

plt.show()

# Features and Target (ML....)

X = df.drop("species", axis=1)

y = df["species"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model

model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:\n")

print(accuracy)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")

print(cm)

# Classification Report

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")

print(report)

# Save model

with open("models/iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully!")