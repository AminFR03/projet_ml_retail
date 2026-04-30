import sys
import pandas as pd
import joblib
import os
import numpy as np

base_dir = os.path.abspath(os.path.join('.', ''))
model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
features_path = os.path.join(base_dir, 'models', 'feature_names.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

# Loyal
input_data_loyal = {
    'Recency': 5,
    'Frequency': 20,
    'MonetaryTotal': 1500,
    'Age': 40
}

# Risky
input_data_risky = {
    'Recency': 250,
    'Frequency': 2,
    'MonetaryTotal': 50,
    'Age': 22
}

def pred_for(input_data):
    df2 = pd.DataFrame([input_data])
    if 'MonetaryTotal' in input_data and 'Recency' in input_data:
        df2['MonetaryPerDay'] = df2['MonetaryTotal'] / (df2['Recency'] + 1)
    if 'MonetaryTotal' in input_data and 'Frequency' in input_data:
        df2['AvgBasketValue'] = df2['MonetaryTotal'] / df2['Frequency']

    scale_cols_list = list(scaler.feature_names_in_)
    for i, col in enumerate(feature_names):
        if col not in df2.columns:
            if col in scale_cols_list:
                df2[col] = scaler.mean_[scale_cols_list.index(col)]
            else:
                df2[col] = 0

    df2 = df2[feature_names]
    df2[scale_cols_list] = scaler.transform(df2[scale_cols_list])

    return model.predict_proba(df2)[0]

print("Loyal:", pred_for(input_data_loyal))
print("Risky:", pred_for(input_data_risky))
