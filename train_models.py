import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_save_models():
    # Load dataset
    try:
        df = pd.read_csv('car_price_dataset_cleaned.csv')
    except Exception:
        df_raw = pd.read_csv('car_price_dataset.csv')
        df = df_raw.copy()
        df['Car_Name'] = df['Car_Name'].astype(str).str.strip()
        df['Fuel_Type'] = df['Fuel_Type'].astype(str).str.strip().str.title()
        df['Fuel_Type'] = df['Fuel_Type'].replace('Nan', np.nan).replace('None', np.nan)
        df['Transmission'] = df['Transmission'].astype(str).str.strip().str.title()
        
        def clean_kms(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).lower().replace('km', '').replace(',', '').strip()
            try:
                return abs(float(val_str))
            except:
                return np.nan
        
        df['Driven_KMs'] = df['Driven_KMs'].apply(clean_kms)
        df = df.drop_duplicates().reset_index(drop=True)
        df['Fuel_Type'] = df['Fuel_Type'].fillna(df['Fuel_Type'].mode()[0])
        df['Engine_Size_L'] = df['Engine_Size_L'].fillna(df['Engine_Size_L'].median())
        df['Mileage_kmpl'] = df['Mileage_kmpl'].fillna(df['Mileage_kmpl'].median())
        df['Driven_KMs'] = df['Driven_KMs'].fillna(df['Driven_KMs'].median())
        df.to_csv('car_price_dataset_cleaned.csv', index=False)

    df['Car_Age'] = 2026 - df['Year']
    df['Brand'] = df['Car_Name'].apply(lambda x: str(x).split()[0])
    df['Model_Name'] = df['Car_Name'].apply(lambda x: " ".join(str(x).split()[1:]) if len(str(x).split()) > 1 else str(x))

    # Features selection matching notebook
    features_df = df[['Fuel_Type', 'Transmission', 'Brand']]
    encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=float)
    encoder.set_output(transform="pandas")
    encoded_df = encoder.fit_transform(features_df)

    numeric_df = df[['Year', 'Driven_KMs', 'Engine_Size_L', 'Mileage_kmpl', 'Car_Age']]
    X = pd.concat([numeric_df, encoded_df], axis=1)
    feature_columns = list(X.columns)

    # Model 1: Linear Regression
    y_reg = df['Selling_Price']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train_r, y_train_r)
    y_pred_r = lr_model.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, y_pred_r)
    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred_r)

    lr_metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    # Model 2: Logistic Regression (Classification)
    median_price = float(df['Selling_Price'].median())
    y_class = (df['Selling_Price'] > median_price).astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_c_scaled, y_train_c)
    y_pred_c = log_model.predict(X_test_c_scaled)

    acc = accuracy_score(y_test_c, y_pred_c)
    cm = confusion_matrix(y_test_c, y_pred_c)
    report = classification_report(y_test_c, y_pred_c, output_dict=True)

    log_metrics = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'median_threshold': median_price,
        'report': report
    }

    # Build unique brand to model mappings
    brand_models = {}
    for brand in sorted(df['Brand'].unique()):
        models = sorted(df[df['Brand'] == brand]['Model_Name'].unique())
        brand_models[brand] = models

    # Save artifacts dictionary
    artifacts = {
        'lr_model': lr_model,
        'log_model': log_model,
        'scaler': scaler,
        'encoder': encoder,
        'feature_columns': feature_columns,
        'median_price': median_price,
        'lr_metrics': lr_metrics,
        'log_metrics': log_metrics,
        'brand_models': brand_models,
        'fuel_types': sorted(df['Fuel_Type'].unique().tolist()),
        'transmissions': sorted(df['Transmission'].unique().tolist())
    }

    joblib.dump(artifacts, 'car_price_models.joblib')
    print("Models and artifacts successfully trained & saved to 'car_price_models.joblib'")

if __name__ == '__main__':
    train_and_save_models()
