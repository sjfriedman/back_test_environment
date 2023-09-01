#imports
import warnings
warnings.filterwarnings("ignore")

import os
import requests

import numpy as np
import pandas as pd

from datetime import timedelta, datetime, time, date

from dotenv import load_dotenv

#pretty print pandas(use head or else full df prints)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

    
#get stock data from alphavantage
def get_data(symbol, start_date, end_date, api_key, redirect_url):
    #iniates holder for dfs
    stock_data_months = []

    #gets months in between months given
    months = pd.period_range(start=start_date, end=end_date, freq='M')

    for month in months:
        #params for alpha
        params = {
            'apikey' : api_key,
            'function': 'TIME_SERIES_INTRADAY',
            'symbol' : symbol,
            'interval' : '1min',
            'month' : month,
            'outputsize' : 'full'
        }

        # Get data
        response = requests.get(redirect_url, params=params)
        data = response.json()

        # Clean data and append to the list
        stock_data = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data.astype(float)
        stock_data.rename(columns={'1. open' : 'open'}, inplace=True)
        stock_data_months.append(stock_data['open'].to_frame(name='price'))

    #return minute to minute prices
    return pd.concat(stock_data_months).sort_index(axis=0)

#determines if the situation indicates to buy/sell/dont know
def trade(stock_data: pd.DataFrame, situation : str, close_situation : str, close_at_end_day : bool):

    #HELPER FUNCTIONS

    #determines if stock moves pct(movement) up or down
    def close_at(start, stock_data : pd.DataFrame, close_situation : str, close_at_end_day : bool):
        #if you want to close at end day get only prices on this date
        if close_at_end_day:
            price = stock_data.loc[stock_data.index.date == start.date()]

        #gets stock data after datetime
        price = (price.loc[(price.index >= start)])
        
        #gets stock data at possible sell points
        movement = price.query(close_situation)['price']
        
        #returns pct change of trade if there is one
        if not movement.empty:
            return pd.Series({'close_time' : movement.index[0], 'pct_change' : ((movement.iloc[0] - price['price'].iloc[0]) / price['price'].iloc[0])})
        
        #returns sale if closed and close is true
        return  pd.Series({'close_time' : price.index[-1], 'pct_change' : ((price['price'].iloc[-1] - price['price'].iloc[0]) / price['price'].iloc[0])})

    #makes sure only one active trade at a time
    def one_at_a_time(trades : pd.DataFrame):
        #iterates through number of rows in trades(this number is the max number of times this function can be called)
        for row_num in range(len(trades)):
            #if either row_num becomes too large for trades since trades changes or if all the shifts are lined up break
            if row_num >= len(trades) or all(trades['open_time'] > trades['close_time'].shift(1)):
                break

            #see if current row shoudl be used
            mask = (trades['open_time'] > trades['close_time'].iloc[row_num]) | (trades.index <= row_num)

            #change trades with answer
            trades = trades[mask].reset_index(drop=True)

        return trades
    
    #REST OF FUNCTION

    #iniates trades df
    trades = pd.DataFrame(columns=['open_time', 'close_time', 'pct_change'])

    #gets the datetime occurences of each "situation"
    trades['open_time'] = pd.Series(stock_data.query(situation).index)

    #performs "trades"
    trades[['close_time', 'pct_change']] = trades['open_time'].apply(close_at, args=(stock_data, close_situation, close_at_end_day))

    #if no trades
    if trades.empty:
        return 'yo strategy foolish -- no buy indicators found'
    
    #get pct_change and cumalative pct_change
    trades['cum_pct_change'] = trades['pct_change'].cumsum()

    #makes sure only one active trade at a time
    trades = one_at_a_time(trades)

    return trades

    #returns accuracy
    # return pd.Series({'occurences' : len(trades), 'profit': trades['pct_change'].sum()})
 
def execute(symbol, buy_query, sell_query, start_date, end_date):
    min_time = '09:30'
    max_time = '16:00'


    #get data from .env file
    load_dotenv()
    api_key = os.getenv("alpha_api_key")
    redirect_url = os.getenv('alpha_redirect_url')

    stock_data = get_data(symbol, start_date, end_date, api_key, redirect_url)

    #gets data during market open not including extra hours(which we can include later)
    stock_data = stock_data.between_time(min_time, max_time)

    #calculations
    stock_data['rolling_10'] = stock_data['price'].rolling(10).mean()
    stock_data['rolling_20'] = stock_data['price'].rolling(20).mean()
    stock_data = stock_data.dropna()
    
    return trade(stock_data, buy_query, sell_query, True)
    

if __name__ == "__main__":
    execute('SPY', 'rolling_10 >= rolling_20', 'rolling_10 <= rolling_20','2020-01','2020-02')