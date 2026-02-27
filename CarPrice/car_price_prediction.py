"""
Car Price Prediction Project
Digital Bhem - Data Science Internship

Run:
python car_price_prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# =========================
# Load Dataset
# =========================

print("Loading dataset...")
data = pd.read_csv("dataset/car_data.csv")

print("Dataset shape:", data.shape)
print(data.head())

# =========================
# Data Cleaning
# =========================

print("\nCleaning dataset...")

# Replace '?' with NaN
data.replace("?", np.nan, inplace=True)

# Convert numeric columns properly
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='ignore')

# Drop missing values
data.dropna(inplace=True)

print("After cleaning, dataset shape:", data.shape)

# Drop unnecessary column if exists
if "car_ID" in data.columns:
    data.drop("car_ID", axis=1, inplace=True)

# =========================
# Encode Categorical Variables
# =========================

print("\nEncoding categorical variables...")

label_encoders = {}

for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# =========================
# Exploratory Data Analysis
# =========================

print("\nGenerating EDA plots...")

# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(data['price'], kde=True)
plt.title("Car Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================
# Prepare Data for Modeling
# =========================

X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Models
# =========================

print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# =========================
# Evaluate Models
# =========================

def evaluate_model(model, name):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\n{name} Results:")
    print("R² Score:", r2)
    print("RMSE:", rmse)
    return r2

r2_lr = evaluate_model(lr, "Linear Regression")
r2_rf = evaluate_model(rf, "Random Forest")

# =========================
# Select Best Model
# =========================

if r2_rf > r2_lr:
    best_model = rf
    print("\nRandom Forest selected as best model.")
else:
    best_model = lr
    print("\nLinear Regression selected as best model.")

# =========================
# Save Model
# =========================

with open("car_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel saved as car_price_model.pkl")

# =========================
# Example Price Prediction
# =========================

sample = X_test.iloc[0:1]
actual_price = y_test.iloc[0]
predicted_price = best_model.predict(sample)[0]

print("\nExample Prediction:")
print("Actual Price:", actual_price)
print("Predicted Price:", predicted_price)

with open("car_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel saved as car_price_model.pkl")

# =========================
# Real-Time Prediction
# =========================

print("\n--- Real-Time Car Price Prediction ---")

engine_size = float(input("Enter engine size: "))
horsepower = float(input("Enter horsepower: "))
city_mpg = float(input("Enter city mpg: "))
highway_mpg = float(input("Enter highway mpg: "))

# Start with average values
input_data = pd.DataFrame([X.mean()])

if "engine-size" in input_data.columns:
    input_data["engine-size"] = engine_size

if "horsepower" in input_data.columns:
    input_data["horsepower"] = horsepower

if "city-mpg" in input_data.columns:
    input_data["city-mpg"] = city_mpg

if "highway-mpg" in input_data.columns:
    input_data["highway-mpg"] = highway_mpg

predicted_price = best_model.predict(input_data)[0]

print("\nEstimated Car Price:", round(predicted_price, 2))