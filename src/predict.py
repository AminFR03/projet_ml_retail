import sys
import pandas as pd
import joblib
import os

def load_resources():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    features_path = os.path.join(base_dir, 'models', 'feature_names.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        print("Model or scaler not found. Please train the model first.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

def predict(input_data, model, scaler, feature_names):
    # Calculate derived features mathematically correctly
    if 'MonetaryTotal' in input_data and 'Recency' in input_data:
        input_data['MonetaryPerDay'] = input_data['MonetaryTotal'] / (input_data['Recency'] + 1)
    if 'MonetaryTotal' in input_data and 'Frequency' in input_data:
        input_data['AvgBasketValue'] = input_data['MonetaryTotal'] / input_data['Frequency'] if input_data['Frequency'] > 0 else 0
        
    df = pd.DataFrame([input_data])
    
    # Ensure all columns match training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_names]
    
    # Scale numerical features using the exact columns the scaler was fitted on
    scale_cols = scaler.feature_names_in_
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    prediction = model.predict(df)
    proba = model.predict_proba(df)
    return prediction[0], proba[0]

def main():
    print("Loading model and resources...")
    model, scaler, feature_names = load_resources()
    
    # Dummy input data
    input_data = {
        'Recency': 10,
        'Frequency': 5,
        'MonetaryTotal': 500,
        'MonetaryPerDay': 50,
        'AvgBasketValue': 100,
        'CustomerTenure': 100,
        'TenureRatio': 0.1
    }
    
    print("Making prediction for input:", input_data)
    pred, proba = predict(input_data, model, scaler, feature_names)
    print(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
    print(f"Probability: {proba}")

if __name__ == '__main__':
    main()
