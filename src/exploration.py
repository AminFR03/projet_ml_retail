import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    """Calculates VIF for numerical columns to identify multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    # Add constant to avoid infinite VIFs if data is not centered
    try:
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    except Exception as e:
        print("Warning: Could not calculate VIF perfectly (might contain NaNs or infinite values).")
        vif_data["VIF"] = np.nan
    return vif_data

def main():
    print("Starting data exploration...")
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load dataset
    df = pd.read_csv('data/raw/dataset.csv')
    print(f"Dataset loaded with shape {df.shape}")
    
    # 2. Data Quality (Missing Values)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Map")
    plt.tight_layout()
    plt.savefig('reports/missing_values.png')
    plt.close()

    # 3. Churn Distribution
    if 'Churn' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Churn')
        plt.title("Churn Distribution")
        plt.savefig('reports/churn_distribution.png')
        plt.close()

    # 4. Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        plt.figure(figsize=(14, 10))
        corr = num_df.corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig('reports/correlation_matrix.png')
        plt.close()
        
        # High correlation > 0.8 log
        high_corr = corr.abs().unstack().sort_values(ascending=False)
        high_corr = high_corr[high_corr >= 0.8]
        high_corr = high_corr[high_corr < 1.0].drop_duplicates()
        if not high_corr.empty:
            print("Highly Correlated Features (|corr| > 0.8):")
            print(high_corr)

    # 5. Multicollinearity (VIF)
    clean_num_df = num_df.dropna()
    if len(clean_num_df) > 10:
        # Scale for VIF
        scaler = StandardScaler()
        scaled_for_vif = pd.DataFrame(scaler.fit_transform(clean_num_df), columns=clean_num_df.columns)
        vif_df = calculate_vif(scaled_for_vif)
        high_vif = vif_df[vif_df['VIF'] > 10].sort_values(by='VIF', ascending=False)
        if not high_vif.empty:
            print("Features with High VIF (>10):")
            print(high_vif)

        # 6. PCA & Explained Variance Plot
        pca = PCA(n_components=min(10, len(clean_num_df.columns)))
        pca.fit(scaled_for_vif)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.tight_layout()
        plt.savefig('reports/pca_explained_variance.png')
        plt.close()
        
        # PCA 2D Plot
        pca_2 = PCA(n_components=2)
        pca_result = pca_2.fit_transform(scaled_for_vif)
        plt.figure(figsize=(8, 6))
        if 'Churn' in df.columns:
            # Match the indices since we dropped NaNs
            churn_labels = df.loc[clean_num_df.index, 'Churn']
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=churn_labels, cmap='coolwarm', alpha=0.6)
            plt.legend(*scatter.legend_elements(), title="Churn")
        else:
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title(f"PCA 2D (Var: {pca_2.explained_variance_ratio_.sum():.2%})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig('reports/pca_2d.png')
        plt.close()

    print("Exploration completed successfully! Visualizations saved in reports/.")

if __name__ == "__main__":
    main()
