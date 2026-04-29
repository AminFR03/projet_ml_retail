import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
import ipaddress
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    return pd.read_csv(filepath)

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def clean_data(df):
    """Handles outliers, missing values, and useless columns."""
    # Outliers
    if 'SupportTicketsCount' in df.columns:
        df.loc[df['SupportTicketsCount'].isin([-1, 99, 999]), 'SupportTicketsCount'] = np.nan
    if 'SatisfactionScore' in df.columns:
        df.loc[df['SatisfactionScore'].isin([-1, 99, 999]), 'SatisfactionScore'] = np.nan
        
    # Drop NewsletterSubscribed
    if 'NewsletterSubscribed' in df.columns:
        df = df.drop(columns=['NewsletterSubscribed'])

    return df

def feature_engineering(df):
    """Engineers new features."""
    # Parsing dates
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
        df['RegYear'] = df['RegistrationDate'].dt.year
        df['RegMonth'] = df['RegistrationDate'].dt.month
        df['RegDay'] = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])

    # Ratios
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = np.where(df['Frequency'] > 0, df['MonetaryTotal'] / df['Frequency'], 0)
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = np.where(df['CustomerTenureDays'] > 0, df['Recency'] / df['CustomerTenureDays'], 0)

    # IP Address handling (Mocking private/public detection)
    if 'LastLoginIP' in df.columns:
        def is_private_ip(ip):
            try:
                return 1 if ipaddress.ip_address(ip).is_private else 0
            except ValueError:
                return 0
        df['IsPrivateIP'] = df['LastLoginIP'].apply(is_private_ip)
        df = df.drop(columns=['LastLoginIP'])
        
    # Drop highly missing features (>50%)
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Drop zero variance features
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:
            df = df.drop(columns=[col])

    return df

def drop_high_correlation(df, threshold=0.8):
    """Drops one of the features from highly correlated pairs."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude Churn from dropping
    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != 'Churn']
    return df.drop(columns=to_drop)

def encode_categorical(X_train, X_test, y_train):
    """Applies specific encoding strategies."""
    # Ordinal or One-Hot: RFMSegment, AgeCategory, SpendingCategory, LoyaltyLevel, ChurnRiskCategory, BasketSizeCategory
    # Ordinal: CustomerType, FavoriteSeason, WeekendPreference, ProductDiversity, Gender, AccountStatus, Region
    # One-Hot: Country
    # Target Encoding: PreferredTimeOfDay

    # Simplified approach for this project:
    one_hot_cols = [c for c in ['Country', 'RFMSegment', 'Gender', 'Region', 'AccountStatus', 'FavoriteSeason', 'CustomerType', 'WeekendPreference', 'ProductDiversity'] if c in X_train.columns]
    ordinal_cols = [c for c in ['AgeCategory', 'SpendingCategory', 'LoyaltyLevel', 'ChurnRiskCategory', 'BasketSizeCategory'] if c in X_train.columns]
    target_cols = [c for c in ['PreferredTimeOfDay'] if c in X_train.columns]

    if one_hot_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_encoded = pd.DataFrame(encoder.fit_transform(X_train[one_hot_cols]), columns=encoder.get_feature_names_out(one_hot_cols), index=X_train.index)
        test_encoded = pd.DataFrame(encoder.transform(X_test[one_hot_cols]), columns=encoder.get_feature_names_out(one_hot_cols), index=X_test.index)
        X_train = pd.concat([X_train.drop(one_hot_cols, axis=1), train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(one_hot_cols, axis=1), test_encoded], axis=1)

    if ordinal_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[ordinal_cols] = encoder.fit_transform(X_train[ordinal_cols])
        X_test[ordinal_cols] = encoder.transform(X_test[ordinal_cols])

    if target_cols:
        encoder = TargetEncoder()
        X_train[target_cols] = encoder.fit_transform(X_train[target_cols], y_train)
        X_test[target_cols] = encoder.transform(X_test[target_cols])
        
    return X_train, X_test

def scale_and_impute(X_train, X_test):
    """Imputes missing values and scales numerical features."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    
    # KNN imputation specifically for Age if present, simple for others
    if 'Age' in num_cols:
        knn = KNNImputer(n_neighbors=5)
        X_train['Age'] = knn.fit_transform(X_train[['Age']])
        X_test['Age'] = knn.transform(X_test[['Age']])
        
    imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler
