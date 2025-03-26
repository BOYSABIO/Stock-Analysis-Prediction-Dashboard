# Stock Trading Dashboard

## Contributors

- Abdallah AlShaarawi  
- Blanca Burgaleta  
- Enrico Tajanlangit  
- Louis-Esmel Kodo  
- Spencer Wood  

---

## Project Overview

This streamlit application is an interactive dashboard that allows:

- Analyzes real-time and historical stock price market trends 
- Provides a sentiment analysis on recent news in the last week about the stock 
- Predicts whether a stock's price will rise or fall the next day (classification)
- Forecasts actual future prices using time series models (regression)
- Simulates investment strategies based on model outputs

It combines ETL pipelines, feature engineering, and ML models using Streamlit.

---

## Project Structure

```
.
├── streamlit_application.py        # Main Streamlit app  
├── streamlit_functions.py         # ML, utility, and plotting functions  
├── ETL_process.py                 # End-to-end data pipeline (Extract, Transform, Enrich)  
├── data/
│   ├── RAW/                       # Raw data from SimFin API  
│   ├── PREPROCESSING/            # Cleaned & merged datasets  
│   ├── ENRICH/                   # Feature-engineered data for modeling  
```

---

## Technologies Used

- Python  
- Streamlit  
- SimFin API (financial statements & stock prices)  
- Finnhub API (news)  
- scikit-learn (logistic regression)  
- TensorFlow/Keras (LSTM model)  
- Plotly (visualizations)  
- pandas / numpy / seaborn / matplotlib  

---

## ETL Pipeline

The full ETL process is implemented in `ETL_process.py`.

### Step 1: Data Ingestion

- Downloads data from SimFin:
  - Stock Prices
  - Income Statements
  - Company Metadata
- Saves to `data/RAW/`

### Step 2: Preprocessing

- Merges price and income data  
- Cleans and fills missing values using:
  - Group-wise fill
  - Median/mode imputation
  - Linear interpolation  
- Drops low-quality columns (e.g., Dividends)  
- Saves cleaned output to `data/PREPROCESSING/`

### Step 3: Feature Engineering

- Adds new features:
  - Price Change %
  - Daily Returns
  - Moving Averages (MA5, MA10)
  - Log Returns
  - Weekday, Month, Quarter  
- Saves to `data/ENRICH/`

---

## Machine Learning Models

### Logistic Regression

- Predicts next-day price direction (up/down)  
- Features: lagged log returns (`Lag1`, `Lag2`)  
- Target: Binary (1 = Up, 0 = Down)  
- Purpose: Generates trading signals  

### LSTM (Long Short-Term Memory)

- Forecasts the actual stock price  
- Uses sequences of past 10 days of price data  
- Outputs:
  - Next-day price
  - 5-day price forecast  
- Generates signal based on comparison with current price  

---

## Trading Strategies

All strategies are backtested and visualized in the dashboard:

### Fixed Units Strategy

- Buys/sells one share per signal  

### Full Volume Strategy

- Invests all cash or sells all shares based on signal  

### Threshold + Sizing Strategy

- Trades a fraction of cash/shares if log return exceeds threshold  
- Scales trade size based on signal strength  

---

## Streamlit Dashboard

The user interface is implemented in `streamlit_application.py`.

### Features

- Select a stock and view:
  - Historical prices and technicals  
  - News headlines and sentiment score  
  - LSTM forecasts  
  - Strategy performance  

To run the dashboard:

```bash
streamlit run streamlit_application.py
```

---

## Environment Variables

Create a `.env` file in your root directory and include:

```
API_KEY=your_simfin_api_key
FINNHUB_API_KEY=your_finnhub_api_key
```

---

## Disclaimer

This project is for educational and simulation purposes only.  
It does not constitute financial advice. Use at your own risk.

---


