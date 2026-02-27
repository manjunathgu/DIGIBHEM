"""
Unemployment Analysis Project
Digital Bhem - Data Science Internship

Run:
python unemployment_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Dataset
# =========================

print("Loading dataset...")
data = pd.read_csv("dataset/unemployment.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Convert date column
data["date"] = pd.to_datetime(data["date"])
data.sort_values("date", inplace=True)

print("\nSummary Statistics:")
print(data.describe())

# =========================
# 1️⃣ Overall Unemployment Trend
# =========================

plt.figure(figsize=(10,5))
plt.plot(data["date"], data["all"])
plt.title("Overall Unemployment Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 2️⃣ Age Group Comparison
# =========================

plt.figure(figsize=(10,5))
plt.plot(data["date"], data["16-24"], label="16-24")
plt.plot(data["date"], data["25-54"], label="25-54")
plt.plot(data["date"], data["55-64"], label="55-64")
plt.legend()
plt.title("Unemployment Rate by Age Group")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 3️⃣ Education Level Comparison (Latest Data)
# =========================

latest = data.iloc[-1]

education_cols = [
    "white_men_high_school",
    "white_men_some_college",
    "white_men_bachelor's_degree",
    "white_men_advanced_degree"
]

edu_values = [latest[col] for col in education_cols]

plt.figure(figsize=(8,5))
plt.bar(education_cols, edu_values)
plt.title("Unemployment by Education Level (Latest Data)")
plt.xticks(rotation=45)
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")