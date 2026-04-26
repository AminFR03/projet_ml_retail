import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer

def load_data(filepath):
    """Loads dataset from filepath."""
    return pd.read_csv(filepath)

def save_data(df, filepath):
    """Saves dataset to filepath."""
    df.to_csv(filepath, index=False)

def parse_dates(df, date_column):
    """Parses date columns and extracts useful features."""
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
        df['RegYear'] = df[date_column].dt.year
        df['RegMonth'] = df[date_column].dt.month
        df['RegDay'] = df[date_column].dt.day
        df['RegWeekday'] = df[date_column].dt.weekday
    return df

def feature_engineering(df):
    """Creates new features based on existing ones."""
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = np.where(df['Frequency'] > 0, df['MonetaryTotal'] / df['Frequency'], 0)
    if 'Recency' in df.columns and 'CustomerTenure' in df.columns:
        df['TenureRatio'] = np.where(df['CustomerTenure'] > 0, df['Recency'] / df['CustomerTenure'], 0)
    return df

def remove_useless_features(df, cols_to_drop):
    """Removes specified features."""
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

def impute_missing_values(X_train, X_test, num_strategy='median', cat_strategy='most_frequent'):
    """Imputes missing values using SimpleImputer and KNN for specific features if needed."""
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=num_strategy)
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
    return X_train, X_test

def scale_features(X_train, X_test):
    """Scales numerical features using StandardScaler."""
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, scaler

def apply_pca(X_train, X_test, n_components=10):
    """Applies PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    pca_train = pca.fit_transform(X_train[num_cols])
    pca_test = pca.transform(X_test[num_cols])
    
    return pca_train, pca_test, pca
