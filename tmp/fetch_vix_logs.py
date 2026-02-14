"""Fetch VIX kernel logs using Kaggle API directly"""
import os
import requests
import json

# Get API token from environment
api_token = os.environ.get('KAGGLE_API_TOKEN', 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea')

# Kaggle API endpoint
kernel_id = 'bigbigzabuton/gold-vix-1'
base_url = 'https://www.kaggle.com/api/v1'

# Try to get kernel output
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}

# Alternative approach: try to get the kernel metadata first
metadata_url = f'{base_url}/kernels/{kernel_id}'
print(f"Fetching metadata from: {metadata_url}")

response = requests.get(metadata_url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
