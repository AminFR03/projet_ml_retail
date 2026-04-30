import pandas as pd
import numpy as np
import os

def generate_full_dummy_dataset(num_samples=500):
    np.random.seed(42)
    
    data = {
        'CustomerID': np.random.randint(10000, 99999, num_samples),
        'Recency': np.random.randint(0, 400, num_samples),
        'Frequency': np.random.randint(1, 50, num_samples),
        'MonetaryTotal': np.random.uniform(-5000, 15000, num_samples),
        'MonetaryAvg': np.random.uniform(5, 500, num_samples),
        'MonetaryStd': np.random.uniform(0, 500, num_samples),
        'MonetaryMin': np.random.uniform(-5000, 5000, num_samples),
        'MonetaryMax': np.random.uniform(0, 10000, num_samples),
        'TotalQuantity': np.random.randint(-10000, 100000, num_samples),
        'AvgQuantityPerTransaction': np.random.uniform(1, 1000, num_samples),
        'MinQuantity': np.random.randint(-8000, 0, num_samples),
        'MaxQuantity': np.random.randint(1, 8000, num_samples),
        'CustomerTenureDays': np.random.randint(0, 730, num_samples),
        'FirstPurchaseDaysAgo': np.random.randint(0, 730, num_samples),
        'PreferredDayOfWeek': np.random.randint(0, 6, num_samples),
        'PreferredHour': np.random.randint(0, 23, num_samples),
        'PreferredMonth': np.random.randint(1, 12, num_samples),
        'WeekendPurchaseRatio': np.random.uniform(0.0, 1.0, num_samples),
        'AvgDaysBetweenPurchases': np.random.uniform(0, 365, num_samples),
        'UniqueProducts': np.random.randint(1, 1000, num_samples),
        'UniqueDescriptions': np.random.randint(1, 1000, num_samples),
        'AvgProductsPerTransaction': np.random.uniform(1, 100, num_samples),
        'UniqueCountries': np.random.randint(1, 5, num_samples),
        'NegativeQuantityCount': np.random.randint(0, 100, num_samples),
        'ZeroPriceCount': np.random.randint(0, 50, num_samples),
        'CancelledTransactions': np.random.randint(0, 50, num_samples),
        'ReturnRatio': np.random.uniform(0.0, 1.0, num_samples),
        'TotalTransactions': np.random.randint(1, 10000, num_samples),
        'UniqueInvoices': np.random.randint(1, 500, num_samples),
        'AvgLinesPerInvoice': np.random.uniform(1, 100, num_samples),
        'Age': np.random.uniform(18, 81, num_samples),
        'SupportTicketsCount': np.random.uniform(-1, 15, num_samples),
        'SatisfactionScore': np.random.uniform(-1, 5, num_samples),
    }

    # Calculate realistic Churn based on Recency, Frequency, and MonetaryTotal
    # High recency -> higher churn. High frequency/monetary -> lower churn.
    logit = (data['Recency'] - 150) / 50.0 - (data['Frequency'] - 10) / 5.0 - (data['MonetaryTotal'] - 500) / 1000.0
    prob_churn = 1.0 / (1.0 + np.exp(-logit))
    prob_churn = np.clip(prob_churn, 0.05, 0.95)
    data['Churn'] = np.random.binomial(1, prob_churn)

    data.update({
        'RFMSegment': np.random.choice(['Champions', 'Fideles', 'Potentiels', 'Dormants'], num_samples),
        'AgeCategory': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'], num_samples),
        'SpendingCategory': np.random.choice(['Low', 'Medium', 'High', 'VIP'], num_samples),
        'CustomerType': np.random.choice(['Hyperactif', 'Regulier', 'Occasionnel', 'Nouveau', 'Perdu'], num_samples),
        'FavoriteSeason': np.random.choice(['Hiver', 'Printemps', 'Ete', 'Automne'], num_samples),
        'PreferredTimeOfDay': np.random.choice(['Matin', 'Midi', 'Apres-midi', 'Soir', 'Nuit'], num_samples),
        'Region': np.random.choice(['UK', 'Europe_N', 'Europe_S', 'Europe_E', 'Europe_C', 'Asie', 'Autre'], num_samples),
        'LoyaltyLevel': np.random.choice(['Nouveau', 'Jeune', 'Etabli', 'Ancien', 'Inconnu'], num_samples),
        'ChurnRiskCategory': np.random.choice(['Faible', 'Moyen', 'Eleve', 'Critique'], num_samples),
        'WeekendPreference': np.random.choice(['Weekend', 'Semaine', 'Inconnu'], num_samples),
        'BasketSizeCategory': np.random.choice(['Petit', 'Moyen', 'Grand', 'Inconnu'], num_samples),
        'ProductDiversity': np.random.choice(['Specialise', 'Modere', 'Explorateur'], num_samples),
        'Gender': np.random.choice(['M', 'F', 'Unknown'], num_samples),
        'AccountStatus': np.random.choice(['Active', 'Suspended', 'Pending', 'Closed'], num_samples),
        'Country': np.random.choice(['UK', 'France', 'Germany', 'Spain', 'Italy'], num_samples),
        'NewsletterSubscribed': ['Yes'] * num_samples,
        'RegistrationDate': pd.date_range(start='2010-01-01', periods=num_samples).strftime('%d/%m/%y').tolist(),
        'LastLoginIP': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(num_samples)]
    })

    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[np.random.choice(df.index, size=int(num_samples*0.3)), 'Age'] = np.nan
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/dataset.csv', index=False)
    print("Generated full dummy dataset at data/raw/dataset.csv")

if __name__ == '__main__':
    generate_full_dummy_dataset()
