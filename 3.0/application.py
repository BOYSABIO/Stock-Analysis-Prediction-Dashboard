import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import datetime
import finnhub 
import os
from dotenv import load_dotenv
from stock_snapshot import get_stock_snapshot, format_number, format_sign, get_company_description, create_graph_ticker

load_dotenv()
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Stock Trading Dashboard",
    layout="wide"
)

# HEADER SECTION
st.title("Stock Trading Dashboard")
st.markdown("""
Select a stock from the dropdown menu, view historical data, model predictions,
and trading recommendations. You can also explore the underlying model code and insights by selecting the dropdown.
""")

# SIDEBAR
# Stock Selecter 
st.sidebar.header("Select Stock")
predefined_tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "CRWD"]
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

stock_snapshot = get_stock_snapshot(ticker)
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

# PLOTLY CHART WITH DUMMY DATA\
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

# Graph code snippet
with st.expander("See how the graph was made"):
    st.code(GRAPH_CODE, language="python")

# TAB SECTION FOR EXTRA INFORMATION
tab1, tab2, tab3 = st.tabs(["TAB 1", "TAB 2", "TAB ..."])
with tab1:
    st.write("Here is some analysis about EDA1 and maybe a graph")
    with st.expander("See Code"):
        st.code("print('Hello')", language = 'python')
with tab2:
    st.write("Here is some analysis about EDA2 and maybe a graph")
    with st.expander("See Code"):
        st.code("print('Hello')", language = 'python')
with tab3:
    st.write("Here are some metrics or somthing")
    st.metric("MSE", 0.123)
    st.metric("R² Score", 0.89)
    with st.expander("See Code"):
        st.code("print('Hello')", language = 'python')    

# MODEL OUTPUT SECTION (PLACEHOLDER)
st.markdown("### Model Prediction")
st.success("**Prediction:** Price will Rise Tomorrow")
st.metric("Confidence Score", "82%")

# SHOW UNDERLYING CODE (EXPANDER)
with st.expander("See ARIMA Model Code"):
    ARIMA_CODE = '''
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=1)
'''
    st.code(ARIMA_CODE, language="python")

# END OF PAGE
st.markdown("---")