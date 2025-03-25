import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import datetime
import finnhub 
import os
from dotenv import load_dotenv
from stock_snapshot2 import *

load_dotenv()
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Stock Trading Dashboard",
    layout="wide"
)

# TOP-LEVEL NAVIGATION
tab1, tab2 = st.tabs(["📌 Home", "📈 Go Live Page"])

# HOME TAB
with tab1:
    st.title("Welcome to Our Stock Trading Dashboard")

    st.markdown("""
    ### Overview
    This application provides a **machine learning-powered stock trading strategy assistant** that allows users to:
    - Analyze real-time and historical market trends
    - View model-driven buy/sell predictions
    - Simulate different trading strategies

    ### Purpose
    The system is built to **help investors make more informed trading decisions** using predictive analytics and dynamic portfolio strategies.

    ### Meet the Developers
    - 👤 Abdallah AlShaarawi   
    - 👤 Blanca Burgaleta 
    - 👤 Enrico Tajanlangit
    - 👤 Louis-Esmel Kodo
    - 👤 Spencer Wood

    > ⚠️ **Disclaimer:** This dashboard is a simulation tool for educational purposes only. 
    > We are not responsible for any losses incurred by acting on the system's financial predictions or recommendations.

    """)
    
with tab2:
    # HEADER SECTION
    st.title("Stock Trading Dashboard")
    st.markdown("""
    Select a stock from the dropdown menu, view historical data, model predictions,
    and trading recommendations. You can also explore the underlying model code and insights by selecting the dropdown.
    """)

    # SIDEBAR
    # Stock Selecter 
    st.sidebar.header("Select Stock")
    predefined_tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "CRWD", "PG"]
    selected_ticker = st.sidebar.selectbox("Choose a ticker:", predefined_tickers)

    ticker = selected_ticker

    # News & Sentiment Score
    # Wordlist
    positive_words = ["gain", "rise", "growth", "profit", "up", "record", "soar", "good"]
    negative_words = ["fall", "loss", "drop", "lawsuit", "down", "cut", "decline", "crash", "threat"]

    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) 

    st.sidebar.subheader("Latest News Headlines")

    today = datetime.date.today()
    last_week = today - datetime.timedelta(days=7)

    try:
        news = finnhub_client.company_news(ticker, _from=last_week.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))

        if isinstance(news, list) and len(news) > 0:
            pos, neg = 0, 0

            for article in news[:5]:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                full_text = (headline + " " + summary).lower()

                if any(word in full_text for word in negative_words):
                    color = "red"
                    neg += sum(word in full_text for word in negative_words)
                elif any(word in full_text for word in positive_words):
                    color = "green"
                    pos += sum(word in full_text for word in positive_words)
                else:
                    color = "gray"

                st.sidebar.markdown(f":{color}[**{headline}**]")
                st.sidebar.caption(f"{article['source']} | {datetime.datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')}")
                st.sidebar.write("---")

            total_mentions = pos + neg
            if total_mentions > 0:
                sentiment_score = (pos / total_mentions) * 100
                st.sidebar.metric("Custom Sentiment Score", f"{sentiment_score:.1f}% Positive")
            else:
                st.sidebar.info("Sentiment analysis not available.")

        else:
            st.sidebar.info("No recent news found for this stock.")

    except Exception as e:
        st.sidebar.warning(f"Error fetching news: {e}")


    # STOCK SNAPSHOT
    st.markdown("### Stock Snapshot")

    df_prices, stock_snapshot = get_stock_snapshot(ticker)
    stock_information = get_company_description(ticker)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{ticker} - {stock_information['name'].iloc[0]}")
        st.caption("**Exchange**: NASDAQ")
        st.caption(f"**Sector**: {stock_information['sectorName'].iloc[0]}")

    with col2:
        st.metric("Current Price", f"${stock_snapshot['Opening Price'].iloc[0]}", f"+{round(stock_snapshot['% change'].iloc[0], 2)}%")
        st.metric("Change", f"{format_sign(round(stock_snapshot['change'].iloc[0], 2))}", delta_color="normal")

    with col3:
        st.metric("Shares Outstanding", f"{format_number(stock_snapshot['Common Shares Outstanding'].iloc[0])}")
        st.metric("Volume", f"{format_number(stock_snapshot['Trading Volume'].iloc[0])}")

    st.markdown("---")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Previous Close", f"${stock_snapshot['Last Closing Price'].iloc[0]}")
    col5.metric("Lowest", f"${stock_snapshot['Lowest Price'].iloc[0]}")
    col6.metric("Highest", f"${stock_snapshot['Highest Price'].iloc[0]}")
    col7.metric("Dividend Paid", f"{format_sign(stock_snapshot['Dividend Paid'].iloc[0])}")

    # COMPANY INFO
    st.info(f"**Company Description:** {stock_information['companyDescription'].iloc[0]}\n\n**Industry:** {stock_information['industryName'].iloc[0]}\n\n**Employees:** {stock_information['numEmployees'].iloc[0]}\n\n**Market:** {stock_information['market'].iloc[0]}""")

    # PLOTLY CHART
    df = create_graph_ticker(ticker)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-Day MA', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-Day MA', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.5))

    fig.update_layout(yaxis2=dict(title="Volume", overlaying='y', side='right'),legend=dict(orientation="h"))
    fig.update_layout(title=f"Price Movement for {ticker}", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig, use_container_width=True)

    GRAPH_CODE = '''
    def create_graph_ticker(ticker):
        df = pd.read_csv('data/RAW/stock_prices.csv')
        df = df[df['Ticker'] == ticker]
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df['MA20'] = df['Close'].rolling(window=20).mean()  
        df['MA50'] = df['Close'].rolling(window=50).mean()  
        return df

    df = create_graph_ticker(ticker)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-Day MA', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-Day MA', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2'))

    fig.update_layout(yaxis2=dict(title="Volume", overlaying='y', side='right'),legend=dict(orientation="h"))
    fig.update_layout(title=f"Price Movement for {ticker}", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig, use_container_width=True)'''

    # Code snippet
    with st.expander("See how the graph was made"):
        st.code(GRAPH_CODE, language="python")


    # MODEL OUTPUT SECTION (PLACEHOLDER)
    st.markdown("### Model Prediction & Strategy Evaluation")

    # User inputs initial cash
    initial_cash = st.number_input("Enter available cash ($)", value=1000, min_value=0)
    fixed_val, full_val, threshold_val, fig, clf, scaler = evaluate_strategy(ticker, initial_cash)

    # Run next-day prediction
    pred, conf = make_next_day_prediction(df, clf, scaler)
    
    # Extract latest price
    latest_price = df["Close"].iloc[-1] if not df.empty else None


    # RECOMMENDED TRADING STRATEGY
    st.markdown("### Recommended Trading Strategy")
    st.plotly_chart(fig, use_container_width=True)

    col8, col9, col10 = st.columns(3)
    with col8:
        st.markdown('**Full Volume Strategy Recommendation**')
        if pred is not None and latest_price is not None:
            if pred == 1:
                qty = int(initial_cash // latest_price)
                st.write(f"You should **buy {qty} shares** at approx. ${latest_price:.2f} each.")
            else:
                st.write("You should **sell or hold cash** — avoid buying today.")
        else:
            st.warning("Not enough data to generate a trading recommendation.")

    with col9:
        st.markdown('**Fixed Units Strategy Recommendation**')
        if pred is not None and latest_price is not None:
            if pred == 1:
                qty = 1
                st.write(f"You should **buy {qty} share** at approx. ${latest_price:.2f}.")
            else:
                st.write(f"You should sell {qty} share** at approx. ${latest_price:.2f}.")
        else:
            st.warning("Not enough data to generate a trading recommendation.")
    
    with col10:
        st.markdown('**Dynamic Threshold Strategy Recommendation**')
        if pred is not None and latest_price is not None:
            if pred == 1:
                st.write(f"**Buy Signal:** Our model predicts price will rise tomorrow with {conf*100:.2f}% confidence.")
            else:
                st.write(f"**Sell Signal:** Our model predicts price will fall tomorrow with {conf*100:.2f}% confidence.")

    if pred == 1:
        st.success(f"**Buy Signal:** Our model predicts price will rise tomorrow with {conf*100:.2f}% confidence.")
    else:
        st.error(f"**Sell Signal:** Our model predicts price will fall tomorrow with {conf*100:.2f}% confidence.")

    st.markdown('---')


    #LSTM MODEL
    st.markdown("### LSTM Model Prediction")
    predictions, predicted_price, true_prices, model, scalar, series, metrics = lstm_model(df_prices)
    fig2, signal, future_df, final_forecast_price = plot_lstm(df_prices, ticker, predictions, predicted_price, true_prices, model, scalar, series, seq_length = 10, future_days = 5)
    st.plotly_chart(fig2, use_container_width=True)

    n_days = future_df.shape[0]
    if signal == 1:
        st.success(f"**Buy Signal:** Our model predicts the price will rise to **${final_forecast_price:.2f}** in {n_days} days.")
    else:
        st.error(f"**Sell Signal:** Our model predicts the price will fall to **${final_forecast_price:.2f}** in {n_days} days.")

    GRAPH_CODE2 = '''
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

        return fig, signal, future_df, final_forecast_price'''
    with st.expander("See how the graph was made"):
        st.code(GRAPH_CODE2, language="python")

    st.markdown('### Forecasted Prices & Metrics')
    st.dataframe(future_df.set_index('Date'))

    col11, col12, col13 = st.columns(3)
    with col11:
        st.metric("RMSE", round(metrics['RMSE'], 2))
    with col12:
        st.metric("MAE", round(metrics['MAE'], 2))
    with col13:
        st.metric("R2", round(metrics['R2'], 2))

    st.markdown("---")