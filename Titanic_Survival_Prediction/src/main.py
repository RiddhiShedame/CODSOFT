import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("D:\\PROJECTS\\Data_Science\\CODSOFT\\Titanic_Survival_Prediction\\data\\train.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:\n")
print(df.info())

# Check missing values
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nColumn Names:\n")
print(df.columns)

# Survival count plot
sns.countplot(x="Survived", data=df)

plt.title("Survival Count")
plt.show()

# Gender vs Survival
sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Gender vs Survival")
plt.show()

# Passenger Class vs Survival
sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Passenger Class vs Survival")
plt.show()

# Age distribution
sns.histplot(df["Age"], bins=30)

plt.title("Age Distribution")
plt.show()

df.drop(columns=["Cabin"], inplace=True)
# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())
# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
# Convert Embarked column into numerical values
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df.drop(columns=["Name", "Ticket","PassengerId"], inplace=True)

print("\nCleaned Dataset:\n")
print(df.head())

print("\nRemaining Missing Values:\n")
print(df.isnull().sum())

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]



# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0.5, 1.5], ["Not Survived", "Survived"])
plt.yticks([0.5, 1.5], ["Not Survived", "Survived"])

plt.show()


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\n================ RANDOM FOREST MODEL ================\n")
# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
# Train Random Forest
rf_model.fit(X_train, y_train)
# Predictions
rf_pred = rf_model.predict(X_test)
# Random Forest Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred))


rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:\n")
print(rf_cm)

plt.figure(figsize=(6,4))

sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0.5, 1.5], ["Not Survived", "Survived"])
plt.yticks([0.5, 1.5], ["Not Survived", "Survived"])

plt.show()

import pickle

# Save model
with open("models/titanic_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nModel saved successfully!")