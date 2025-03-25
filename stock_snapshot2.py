import requests 
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
    latest = df.tail(2).copy()
    latest['% change'] = latest['Last Closing Price'].pct_change()
    latest['change'] = latest['Last Closing Price'].diff()
    return df, latest.tail(1)

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

def make_next_day_prediction(recent_values_df, clf, scaler):
    ticker_df = recent_values_df.copy()
    ticker_df.set_index("date", inplace=True)
    ticker_df["Log_Returns"] = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1))
    ticker_df["Lag1"] = ticker_df["Log_Returns"].shift(1)
    ticker_df["Lag2"] = ticker_df["Log_Returns"].shift(2)
    
    # Use only the most recent row with all lag features
    latest = ticker_df.dropna(subset=["Lag1", "Lag2"]).iloc[-1:]
    if latest.empty:
        return None, None  # Not enough data
    
    X_latest = latest[["Lag1", "Lag2"]]
    X_scaled = scaler.transform(X_latest)
    
    prediction = clf.predict(X_scaled)[0]
    confidence = clf.predict_proba(X_scaled)[0][prediction]

    return prediction, confidence


# 2. Utility Functions
def prepare_data(ticker):
    """Extracts, transforms, and returns data for one ticker."""
    # Load data (edit path if needed)
    df = pd.read_csv("data/ENRICH/merged_stock_income.csv", parse_dates=["Date"], usecols=["Date", "Ticker", "Close"],low_memory=True) #for saving memory
    df.set_index("Date", inplace=True)
    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df["Log_Returns"] = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1))
    ticker_df["Lag1"] = ticker_df["Log_Returns"].shift(1)
    ticker_df["Lag2"] = ticker_df["Log_Returns"].shift(2)
    ticker_df["Target"] = (ticker_df["Close"].shift(-1) > ticker_df["Close"]).astype(int)
    return ticker_df.dropna(subset=["Lag1", "Lag2", "Target"])

def train_model(model_data):
    X = model_data[["Lag1", "Lag2"]]
    y = model_data["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf, scaler, X_test, y_test

# 3. Strategy Functions
def strategy_fixed_units(prices, predictions, initial_cash=1000):
    cash, shares = initial_cash, 0
    history = []
    for pred, price in zip(predictions, prices):
        if pred == 1 and cash >= price:
            shares += 1
            cash -= price
        elif pred == 0 and shares > 0:
            shares -= 1
            cash += price
        history.append(cash + shares * price)
    return history

def strategy_full_volume(prices, predictions, initial_cash=1000):
    cash, shares = initial_cash, 0
    history = []
    for pred, price in zip(predictions, prices):
        if pred == 1 and cash >= price:
            qty = int(cash // price)
            shares += qty
            cash -= qty * price
        elif pred == 0 and shares > 0:
            cash += shares * price
            shares = 0
        history.append(cash + shares * price)
    return history

def strategy_threshold_dynamic(data, model, scaler, test_index, investment_fraction=0.2, threshold=0.005, initial_cash=1000):
    data = data.copy()
    cash = initial_cash
    shares = 0
    history = []

    for i in range(2, len(data)):
        if data.index[i] not in test_index:
            continue
        row = data.iloc[i - 1][["Lag1", "Lag2"]].to_frame().T
        log_return = data.iloc[i]["Log_Returns"]
        scaled = scaler.transform(row)
        prediction = model.predict(scaled)[0]
        price = data.iloc[i]["Close"]
        # print(f"Day {i}, Log Return: {log_return:.4f}, Prediction: {prediction}, Cash: {cash:.2f}, Shares: {shares}")

        if log_return > threshold and cash >= price:
            invest_amt = cash * investment_fraction
            qty = int(invest_amt // price)
            shares += qty
            cash -= qty * price
        elif log_return < -threshold and shares > 0:
            sell_qty = int(shares * investment_fraction)
            shares -= sell_qty
            cash += sell_qty * price

        portfolio_value = cash + shares * price
        history.append(portfolio_value)

    return history


# 4. Backtest and Visualization
def evaluate_strategy(ticker, initial_cash=1000):
    data = prepare_data(ticker)
    clf, scaler, X_test, y_test = train_model(data)
    y_pred = clf.predict(X_test)
    prices = data.loc[y_test.index, "Close"].values

    # Pass user-defined cash to strategies
    fixed = strategy_fixed_units(prices, y_pred, initial_cash=initial_cash)
    full = strategy_full_volume(prices, y_pred, initial_cash=initial_cash)
    thresholded = strategy_threshold_dynamic(data, clf, scaler, y_test.index, initial_cash=initial_cash)


    # Plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=fixed, mode='lines', name='Fixed Units'
    ))
    fig.add_trace(go.Scatter(
        y=full, mode='lines', name='Full Volume'
    ))
    fig.add_trace(go.Scatter(
        y=thresholded, mode='lines', name='Threshold + Sizing'
    ))

    fig.update_layout(
        title=f"{ticker} - Portfolio Value Comparison",
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",  # Optional: matches your Streamlit theme
        legend_title="Strategy"
    )

    return fixed[-1], full[-1], thresholded[-1], fig, clf, scaler




# 5. Multi-Ticker Comparison
def multi_ticker_comparison(list_of_tickers):
    tickers = list_of_tickers
    results = {}

    for t in tickers:
        print(f"\nEvaluating: {t}")
        fixed_val, full_val, thresh_val = evaluate_strategy(t)
        results[t] = {
            "Fixed Units": fixed_val,
            "Full Volume": full_val,
            "Threshold + Sizing": thresh_val
        }

    # Final comparison
    summary_df = pd.DataFrame(results).T
    summary_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Final Portfolio Value Comparison across Strategies")
    plt.ylabel("Final Portfolio Value ($)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    return summary_df


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def lstm_model(df_prices):
    df = df_prices.copy()
    df = df[['Date', 'Last Closing Price']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Last Closing Price']]
    series = data['Last Closing Price'].dropna().values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    train_size = int(len(scaled_series) * 0.8)
    train, test = scaled_series[:train_size], scaled_series[train_size:]

    seq_length = 10
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
    mae = mean_absolute_error(true_prices, predicted_prices)
    r2 = r2_score(true_prices, predicted_prices)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return predictions, predicted_prices, true_prices, model, scaler, series, metrics


def plot_lstm(df_prices, ticker, predictions, predicted_prices, true_prices, model, scaler, full_series, seq_length=10, future_days=5):
    # Predict Future Prices
    last_sequence = full_series[-seq_length:]
    last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    input_seq = last_scaled.reshape(1, seq_length, 1)

    future_preds_scaled = []

    for i in range(future_days):
        next_scaled = model.predict(input_seq, verbose=0)
        future_preds_scaled.append(next_scaled[0])
        input_seq = np.append(input_seq[:, 1:, :], [[next_scaled[0]]], axis=1)

    future_preds = scaler.inverse_transform(future_preds_scaled)

    df = df_prices.copy()
    df = df[['Date', 'Last Closing Price']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    historical_dates = df['Date'].values

    future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=future_days)

    # Plotly
    fig = go.Figure()

    # Original
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=full_series.flatten(),
        mode='lines',
        name='Historical Prices'
    ))

    # Predicted Test
    start_idx = len(historical_dates) - len(true_prices)
    fig.add_trace(go.Scatter(
        x=historical_dates[start_idx:],
        y=predicted_prices.flatten(),
        mode='lines',
        name='LSTM Predictions'
    ))

    # Predicted Future
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds.flatten(),
        mode='lines+markers',
        name=f'{future_days}-Day Forecast',
        line=dict(dash='dash')
    ))

    # Layout
    fig.update_layout(
        title=f'{ticker} LSTM Forecast with {future_days}-Day Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified'
    )

    # If forecast higher than current then buy signal and vice versa
    last_price = full_series[-1][0] 
    final_forecast = future_preds[-1][0]

    if final_forecast > last_price:
        signal = 1  
    else:
        signal = 0 

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_preds.flatten()
    })

    final_forecast_price = future_df['Predicted Price'].iloc[-1]

    return fig, signal, future_df, final_forecast_price