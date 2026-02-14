"""Scrape Kaggle kernel page to get error log"""
import requests
from bs4 import BeautifulSoup
import json
import os

# Kernel URL
kernel_url = 'https://www.kaggle.com/code/bigbigzabuton/gold-vix-1'

# Try to get the page
print(f"Fetching: {kernel_url}")
response = requests.get(kernel_url)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Try to find error message or log data in script tags
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'error' in script.string.lower():
            print("\n=== Found error in script tag ===")
            print(script.string[:1000])
            print("\n...")

    # Try to find any visible error messages
    errors = soup.find_all(text=lambda text: text and 'error' in text.lower())
    if errors:
        print("\n=== Found error text ===")
        for err in errors[:5]:
            print(err.strip())

    # Save the raw HTML for manual inspection
    with open('tmp/vix_v6_page.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("\n=== Saved page to tmp/vix_v6_page.html ===")
else:
    print(f"Failed to fetch page: {response.status_code}")
