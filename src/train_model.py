import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_classification(X_train, y_train, X_test, y_test):
    print("\n--- CLASSIFICATION (Predicting Churn) ---")
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    print(f"LR - Accuracy: {accuracy_score(y_test, lr_preds):.3f}")
    print(f"LR - F1 Score: {f1_score(y_test, lr_preds):.3f}")
    print(f"LR - AUC: {roc_auc_score(y_test, lr_probs):.3f}")

    # 2. Random Forest with GridSearchCV
    print("Training Random Forest with GridSearchCV...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    rf_preds = best_rf.predict(X_test)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    
    print(f"Best RF Params: {grid.best_params_}")
    print(f"RF - Accuracy: {accuracy_score(y_test, rf_preds):.3f}")
    print(f"RF - F1 Score: {f1_score(y_test, rf_preds):.3f}")
    print(f"RF - AUC: {roc_auc_score(y_test, rf_probs):.3f}")
    
    # Save best model (assuming RF is better for complex datasets)
    joblib.dump(best_rf, 'models/best_model.pkl')
    joblib.dump(list(X_train.columns), 'models/feature_names.pkl')
    print("Classification model saved to models/best_model.pkl")

def train_regression(X_train, X_test):
    print("\n--- REGRESSION (Predicting MonetaryTotal) ---")
    # We need MonetaryTotal in the original data to predict it. 
    # Since preprocessing removes the target, we should extract MonetaryTotal from the raw dataset, 
    # but for simplicity, we will assume MonetaryTotal is still in X_train (we didn't drop it unless high corr).
    if 'MonetaryTotal' not in X_train.columns:
        print("MonetaryTotal not found in features. Skipping regression.")
        return
        
    y_reg_train = X_train['MonetaryTotal']
    X_reg_train = X_train.drop(columns=['MonetaryTotal'])
    
    y_reg_test = X_test['MonetaryTotal']
    X_reg_test = X_test.drop(columns=['MonetaryTotal'])
    
    print("Training Random Forest Regressor...")
    rfr = RandomForestRegressor(n_estimators=50, random_state=42)
    rfr.fit(X_reg_train, y_reg_train)
    preds = rfr.predict(X_reg_test)
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, preds))
    r2 = r2_score(y_reg_test, preds)
    
    print(f"Regression RMSE: {rmse:.3f}")
    print(f"Regression R2: {r2:.3f}")
    joblib.dump(rfr, 'models/regression_model.pkl')
    
    # Generate and save scatter plot for regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_reg_test, preds, alpha=0.5, color='blue')
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.xlabel('Valeurs Réelles (MonetaryTotal)')
    plt.ylabel('Valeurs Prédites')
    plt.title(f'Régression : Prédit vs Réel (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig('reports/regression_scatter.png')
    plt.close()
    print("Regression scatter plot saved to reports/regression_scatter.png")

def train_clustering(X_train):
    print("\n--- CLUSTERING (K-Means on RFM logic) ---")
    # We look for Recency, Frequency, Monetary variables
    rfm_cols = [c for c in X_train.columns if any(x in c for x in ['Recency', 'Frequency', 'Monetary'])]
    if len(rfm_cols) < 2:
        print("Not enough RFM columns for clustering. Using all features.")
        rfm_cols = X_train.columns
        
    X_rfm = X_train[rfm_cols]
    
    # Elbow method
    inertias = []
    silhouettes = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        preds = kmeans.fit_predict(X_rfm)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_rfm, preds))
        
    # Find best K based on silhouette
    best_k = K_range[np.argmax(silhouettes)]
    print(f"Optimal number of clusters (based on Silhouette): {best_k}")
    
    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    best_kmeans.fit(X_rfm)
    joblib.dump(best_kmeans, 'models/kmeans_model.pkl')
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouettes, marker='o', color='orange')
    plt.title('Silhouette Scores')
    
    plt.tight_layout()
    plt.savefig('reports/kmeans_metrics.png')
    plt.close()
    print("Clustering model saved. Metrics saved to reports/kmeans_metrics.png")

def main():
    print("Loading data...")
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_test = pd.read_csv('data/train_test/X_test.csv')
    y_train = pd.read_csv('data/train_test/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/train_test/y_test.csv').values.ravel()

    train_classification(X_train, y_train, X_test, y_test)
    train_regression(X_train, X_test)
    train_clustering(X_train)
    
    print("\nAll modeling tasks completed successfully!")

if __name__ == "__main__":
    main()
