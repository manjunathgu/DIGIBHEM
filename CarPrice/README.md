# 🚗 Car Price Prediction

## 📌 Project Overview
This project predicts car prices using Machine Learning regression models.
It was developed as part of the Digital Bhem Data Science Internship.

The model analyzes various car features such as engine size, horsepower, mileage, and other specifications to estimate car price accurately.

---

## 📂 Dataset
- Source: Kaggle Automobile Dataset
- Total Records: 205
- Total Features: 26
- Target Variable: price

---

## 🧹 Data Preprocessing
- Replaced "?" values with NaN
- Converted numeric columns correctly
- Dropped missing values
- Encoded categorical variables using Label Encoding
- Removed unnecessary columns if present

---

## 📊 Exploratory Data Analysis
- Price distribution plot
- Correlation heatmap
- Feature relationship analysis

---

## 🤖 Models Used
1. Linear Regression
2. Random Forest Regressor

---

## 📈 Model Evaluation
Models were evaluated using:
- R² Score
- Root Mean Squared Error (RMSE)

The best model is automatically selected based on R² performance.

---

## 💾 Model Saving
The trained best model is saved as:
car_price_model.pkl

---

## 🔮 Prediction Capabilities
- Example prediction using test dataset
- Real-time prediction using user input:
  - Engine Size
  - Horsepower
  - City MPG
  - Highway MPG

---

## ▶️ How to Run the Project

1. Activate virtual environment:
   .\.venv\Scripts\activate

2. Navigate to CarPrice folder:
   cd CarPrice

3. Install dependencies:
   pip install -r requirements.txt

4. Run the script:
   python car_price_prediction.py

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🎯 Internship Requirements Completed
✔ Data Cleaning  
✔ Exploratory Data Analysis  
✔ Regression Modeling  
✔ Model Evaluation  
✔ Model Saving  
✔ Real-Time Prediction  

---

Developed for Digital Bhem Data Science Internship