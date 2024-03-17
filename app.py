import streamlit as st
import requests
import pandas as pd
from prophet import Prophet
import yfinance as yf
import datetime as dt

def main():
    st.title('Welcome to CryptoCurrency Price Prediction')
    df = get_coin()
    coin = st.selectbox('Coins', df,help='Select the coin from the dropdown to get the prediction of that coin')
    days = st.slider('Days to Predict', 1, 365, 1)
    st.write(f'''You have select the coin as {coin} and Days to predict is {days}''')
    prediction = train_predict(coin,days)
    st.write(f'''# The Prediction of {coin} for the next {days} day from today is {prediction}''')


def train_predict(coin_name,days):
    df = yf.download(f'{coin_name}-USD',dt.date.today() - dt.timedelta(days=365),dt.date.today())
    df.reset_index(inplace=True)
    df = df[['Date','Adj Close']]
    df.columns = ['ds','y']
    model = Prophet()
    model.fit(df)
    future_dates = model.make_future_dataframe(periods = days)
    prediction = model.predict(future_dates)
    prediction = prediction[['ds','yhat']]
    return round(prediction['yhat'].iloc[-1],2)


@st.cache_data
def get_coin():
    data = requests.get('https://api.coingecko.com/api/v3/coins/markets', params={"vs_currency": "usd"}).json()
    df = pd.DataFrame(data)
    df = df['symbol']
    return df

def get_coin_detail():
    data = requests.get('https://api.coingecko.com/api/v3/coins/markets', params={"vs_currency": "usd"}).json()
    df = pd.DataFrame(data)
    df = df[['market_cap_rank','name','symbol','current_price','market_cap','circulating_supply']]
    df.set_index('market_cap_rank',inplace=True)
    st.title('Real time crypto-price')
    st.table(df)


st.set_page_config(
   page_title="Crypto Price Prediction",
   page_icon="🤑",
   layout="wide",
   initial_sidebar_state="expanded",
)

menu_option = st.sidebar.selectbox('menu',['Real Time Price','Prediction'], label_visibility='hidden')

if menu_option == 'Real Time Price':
    get_coin_detail()
else:
    main()
