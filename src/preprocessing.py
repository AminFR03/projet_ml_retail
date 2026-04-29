import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import utils
import joblib

def main():
    print("Starting preprocessing...")
    raw_data_path = 'data/raw/dataset.csv'
    if not os.path.exists(raw_data_path):
        print(f"Warning: {raw_data_path} not found. Please run src/generate_dummy_data.py first.")
        return

    df = utils.load_data(raw_data_path)

    # 1. Cleaning & Outliers
    print("Cleaning data...")
    df = utils.clean_data(df)

    # 2. Feature Engineering
    print("Engineering features...")
    df = utils.feature_engineering(df)

    # Define Target and Features
    target = 'Churn'
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found!")
        return

    # Drop CustomerID as it's useless for modeling
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # 3. Handle high correlation
    print("Removing highly correlated features...")
    df = utils.drop_high_correlation(df, threshold=0.8)

    X = df.drop(columns=[target])
    y = df[target]

    # 4. Train/Test Split (Stratified on Churn)
    print("Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Encoding Categorical
    print("Encoding categorical features...")
    X_train, X_test = utils.encode_categorical(X_train, X_test, y_train)

    # 6. Scaling & Imputation
    print("Scaling and imputing missing values...")
    X_train, X_test, scaler = utils.scale_and_impute(X_train, X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 7. Class Imbalance Handling
    print("Applying SMOTE to balance classes on training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Save to train_test folder
    print("Saving processed data...")
    os.makedirs('data/train_test', exist_ok=True)
    utils.save_data(X_train_res, 'data/train_test/X_train.csv')
    utils.save_data(X_test, 'data/train_test/X_test.csv')
    utils.save_data(pd.DataFrame(y_train_res, columns=[target]), 'data/train_test/y_train.csv')
    utils.save_data(pd.DataFrame(y_test, columns=[target]), 'data/train_test/y_test.csv')
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
