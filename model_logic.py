import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def get_trained_model():
    # 1. Load the Excel file
    file_name = 'GYTS4.xls'
    try:
        # Using xlrd engine for .xls files
        df = pd.read_excel(file_name)
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

    # 2. Clean Column Names (removes hidden spaces)
    df.columns = df.columns.astype(str).str.strip()

    # 3. Preprocessing
    # Remove aggregate 'India' row for training, but keep it for national stats later
    df_clean = df[df['State/UT'] != 'India'].copy()
    
    def clean_val(x):
        if pd.isna(x) or str(x).strip() == '--':
            return np.nan
        x = str(x).replace('<', '').replace('>', '').replace('%', '').strip()
        try:
            return float(x)
        except:
            return np.nan

    for col in df_clean.columns:
        if col not in ['State/UT', 'Area']:
            df_clean[col] = df_clean[col].apply(clean_val)

    # 4. Target & Feature Selection
    target = 'Current tobacco users (%)'
    to_drop = ['State/UT', 'Area', target, 'Ever tobacco users (%)', 
               'Current tobacco smokers (%)', 'Current smokeless tobacco users (%)']
    
    X = df_clean.drop(columns=[c for c in to_drop if c in df_clean.columns])
    y = df_clean[target]

    # Handle missing values
    mask = ~y.isna()
    X = X[mask].fillna(X.median())
    y = y[mask]

    # 5. AI Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    r2 = r2_score(y, model.predict(X))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, importances, r2, df_clean