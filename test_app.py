import sys
import os
sys.path.append(os.path.abspath('app'))
from app import app
import json

client = app.test_client()

print("Testing Loyal:")
res_loyal = client.post('/predict', json={
    'Recency': 5,
    'Frequency': 20,
    'MonetaryTotal': 1500,
    'Age': 40
})
print(res_loyal.get_json())

print("Testing Risky:")
res_risky = client.post('/predict', json={
    'Recency': 250,
    'Frequency': 2,
    'MonetaryTotal': 50,
    'Age': 22
})
print(res_risky.get_json())
