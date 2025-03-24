import requests 
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
load_dotenv()
API_KEY = os.getenv('API_KEY')

def get_stock_snapshot(ticker):
    url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}"
    headers = {
        "accept": "application/json",
        "Authorization": API_KEY
    }
    response = requests.get(url, headers=headers).json()
    json_data = response[0]
    df = pd.DataFrame(json_data['data'], columns=json_data['columns'])
    latest = df.tail(2)
    latest['% change'] = latest['Last Closing Price'].pct_change()
    latest['change'] = latest['Last Closing Price'].diff()
    return latest.tail(1)

def format_number(num):
    if num >= 1000000000000: 
        return f"{num / 1000000000000:.1f}T"
    elif num >= 1000000000:
        return f"{num / 1000000000:.1f}B"
    elif num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}K"
    else: 
        return str(num)
    
def format_sign(num):
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return 'None'
    elif num > 0:
        return f"+${num}"
    else:
        return f"-${abs(num)}"
    
def get_company_description(ticker):
    url = f"https://backend.simfin.com/api/v3/companies/general/compact?ticker={ticker}"

    headers = {
        "accept": "application/json",
        "Authorization": API_KEY
    }

    response = requests.get(url, headers=headers).json()
    df = pd.DataFrame(response['data'], columns=response['columns'])
    return df

def create_graph_ticker(ticker):
    df = pd.read_csv('data/RAW/stock_prices.csv')
    df = df[df['Ticker'] == ticker]
    df['date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['MA20'] = df['Close'].rolling(window=20).mean()  
    df['MA50'] = df['Close'].rolling(window=50).mean()  
    return df