import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset

df = pd.read_csv("data/advertising.csv")

# Show first 5 rows

print(df.head())
print("\nDataset Info:\n")

print(df.info())

print("\nMissing Values:\n")

print(df.isnull().sum())

print("\nStatistical Summary:\n")

print(df.describe())

# Correlation Heatmap

plt.figure(figsize=(8, 5))

sns.heatmap(df.corr(), annot=True, cmap="Blues")

plt.title("Correlation Heatmap")

plt.show()

# Features and Target

X = df.drop("Sales", axis=1)

y = df["Sales"]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)

print("\nMean Squared Error:", mse)

print("\nR2 Score:", r2)

# Actual vs Predicted Plot

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# Save model

with open("models/sales_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully!")