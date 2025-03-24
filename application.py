import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import datetime
import finnhub 
import os
from dotenv import load_dotenv

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

# STOCK SELECTOR SECTION
st.sidebar.header("Select Stock")
predefined_tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
selected_ticker = st.sidebar.selectbox("Choose a ticker:", predefined_tickers)

ticker = selected_ticker


# STOCK SNAPSHOT
st.markdown("### Stock Snapshot")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader(f"{ticker} - Example Corp")
    st.caption("**Exchange**: NASDAQ")
    st.caption("**Sector**: Technology")

with col2:
    st.metric("Current Price", "$391.50", "+0.06%")
    st.metric("Change", "+$0.24", delta_color="normal")

with col3:
    st.metric("Bid", "$391.40")
    st.metric("Ask", "$391.72")

st.markdown("---")

col4, col5, col6, col7 = st.columns(4)
col4.metric("Previous Close", "$386.84")
col5.metric("Open", "$383.22")
col6.metric("Volume", "39.7M")
col7.metric("Day Range", "$382.80 - $391.74")


# NEWS AND SENTIMENT SCORE
import finnhub
import datetime

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)  # Or hardcoded key for testing

st.sidebar.subheader("Latest News Headlines")

today = datetime.date.today()
last_week = today - datetime.timedelta(days=7)

try:
    news = finnhub_client.company_news(ticker, _from=last_week.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))

    if isinstance(news, list) and len(news) > 0:
        for article in news[:5]:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            sentiment = article.get("sentiment", "neutral")  # Optional: default fallback

            # Sentiment coloring (mock logic for now, Finnhub doesn’t give direct sentiment for news)
            if "crash" in summary.lower() or "lawsuit" in summary.lower() or "down" in summary.lower() or 'threat' in summary.lower():
                color = "red"
            elif "growth" in summary.lower() or "profit" in summary.lower() or "up" in summary.lower():
                color = "green"
            else:
                color = "gray"

            st.sidebar.markdown(f":{color}[**{headline}**]")
            st.sidebar.caption(f"{article['source']} | {datetime.datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')}")
            st.sidebar.write("---")
    else:
        st.sidebar.info("No recent news found for this stock.")

except Exception as e:
    st.sidebar.warning(f"Error fetching news: {e}")

positive_words = ["gain", "rise", "growth", "profit", "up", "record", "soar"]
negative_words = ["fall", "loss", "drop", "lawsuit", "down", "cut", "decline"]

pos, neg = 0, 0
for article in news:
    text = (article.get("headline", "") + " " + article.get("summary", "")).lower()
    pos += sum(word in text for word in positive_words)
    neg += sum(word in text for word in negative_words)

if pos + neg > 0:
    sentiment_score = (pos / (pos + neg)) * 100
    st.sidebar.metric("Custom Sentiment Score", f"{sentiment_score:.1f}% Positive")
else:
    st.sidebar.info("Sentiment analysis not available.")


# COMPANY INFO 
st.info(f"**Company Name:** Example Corp\n\n**Industry:** Technology\n\n**Employees:** 10,000\n\n**P/E Ratio:** 24.5")

# DUMMY DATA (REPLACE WITH REAL ETL/API CALL)
dates = pd.date_range(start="2024-01-01", periods=60)
prices = pd.Series([100 + i + (i%5)*3 for i in range(60)])
data = pd.DataFrame({"Date": dates, "Price": prices})

# PLOTLY CHART SECTION
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Price"], mode='lines+markers', name='Price'))
fig.update_layout(title=f"Price Movement for {ticker}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

GRAPH_CODE = '''
# DUMMY DATA (REPLACE WITH REAL ETL/API CALL)
dates = pd.date_range(start="2024-01-01", periods=60)
prices = pd.Series([100 + i + (i%5)*3 for i in range(60)])
data = pd.DataFrame({"Date": dates, "Price": prices})

# PLOTLY CHART SECTION
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Price"], mode='lines+markers', name='Price'))
fig.update_layout(title=f"Price Movement for {ticker}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)'''

# Graph code snippet
with st.expander("See how the graph was made"):
    st.code(GRAPH_CODE, language="python")

tab1, tab2, tab3 = st.tabs(["EDA 1", "EDA 2", "EDA ..."])
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
with st.expander("🔍 See ARIMA Model Code"):
    ARIMA_CODE = '''
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=1)
'''
    st.code(ARIMA_CODE, language="python")

# OPTIONAL FOOTER / NEXT STEPS
st.markdown("---")
st.markdown("Feel free to switch stocks or check different models from the sidebar. More features coming soon!")


