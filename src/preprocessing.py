import os
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
import joblib

def main():
    print("Starting preprocessing...")
    # Define paths
    raw_data_path = 'data/raw/dataset.csv' # Placeholder filename, user needs to adjust this
    train_x_path = 'data/train_test/X_train.csv'
    test_x_path = 'data/train_test/X_test.csv'
    train_y_path = 'data/train_test/y_train.csv'
    test_y_path = 'data/train_test/y_test.csv'

    if not os.path.exists(raw_data_path):
        print(f"Warning: {raw_data_path} not found. Please place your dataset in data/raw/ and rename it to dataset.csv or update this script.")
        # Create a dummy dataset for demonstration if not exists
        print("Creating a dummy dataset to demonstrate the pipeline...")
        os.makedirs('data/raw', exist_ok=True)
        df = pd.DataFrame({
            'CustomerID': range(100),
            'Recency': range(100),
            'Frequency': range(1, 101),
            'MonetaryTotal': [x*10 for x in range(100)],
            'CustomerTenure': range(1, 101),
            'RegistrationDate': ['12/03/10'] * 100,
            'NewsletterSubscribed': ['Yes'] * 100,
            'Gender': ['M', 'F'] * 50,
            'Churn': [0, 1] * 50
        })
        df.to_csv(raw_data_path, index=False)
        print("Dummy dataset created at data/raw/dataset.csv")

    df = utils.load_data(raw_data_path)

    # 1. Parsing dates
    print("Parsing dates...")
    df = utils.parse_dates(df, 'RegistrationDate')

    # 2. Feature engineering
    print("Engineering features...")
    df = utils.feature_engineering(df)

    # 3. Remove useless features
    print("Removing useless features...")
    features_to_drop = ['NewsletterSubscribed', 'LastLoginIP', 'RegistrationDate', 'CustomerID']
    df = utils.remove_useless_features(df, features_to_drop)

    # Define Target and Features
    target = 'Churn'
    if target not in df.columns:
        print(f"Warning: target column '{target}' not found in dataset. Using last column as target.")
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoding for categorical variables before splitting
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 4. Train/Test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Imputation
    print("Imputing missing values...")
    X_train, X_test = utils.impute_missing_values(X_train, X_test)

    # 6. Scaling
    print("Scaling features...")
    X_train, X_test, scaler = utils.scale_features(X_train, X_test)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # Save to train_test folder
    print("Saving processed data...")
    os.makedirs('data/train_test', exist_ok=True)
    utils.save_data(X_train, train_x_path)
    utils.save_data(X_test, test_x_path)
    utils.save_data(pd.DataFrame(y_train), train_y_path)
    utils.save_data(pd.DataFrame(y_test), test_y_path)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
