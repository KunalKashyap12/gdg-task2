# 🚗 Used Car Price & Category Predictor (Streamlit Web App)

A machine learning web application built with **Streamlit**, featuring both **Linear Regression** (Continuous Resale Price prediction) and **Logistic Regression** (High vs Budget Price classification), complete with interactive EDA and dataset assessment.

---

## 🚀 Features

- **🎯 Interactive Predictor Frontend**: Select Brand, Model, Year, Driven KMs, Fuel Type, Transmission, Engine Size, and Mileage to get live resale price estimates and high-tier classification confidence.
- **📊 Exploratory Data Analysis (EDA)**: Price distribution, scatter plots, box plots by fuel type, and correlation heatmap.
- **📈 ML Performance Dashboard**: Displays MAE, RMSE, R² Score for Linear Regression, and Accuracy Score & Confusion Matrix for Logistic Regression.
- **📋 Dataset Viewer**: Inspect raw vs cleaned dataset records and summary statistics.

---

## 💻 Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & Generate Models**:
   ```bash
   python train_models.py
   ```

3. **Launch the Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---
