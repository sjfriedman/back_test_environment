#imports
import warnings
warnings.filterwarnings("ignore")

import os
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats

from datetime import timedelta, datetime, time, date

from dotenv import load_dotenv

# # #pretty print pandas(use head or else full df prints)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#get stock data from alphavantage
def get_data(symbol : str, api_key : str, redirect_url : str):

    #set parameter values for alphavantage
    params = {
        'apikey' : api_key,
        'function': 'TIME_SERIES_INTRADAY',
        'symbol' : symbol,
        'interval' : '1min',
        'month' : '2020-12',
        'outputsize' : 'full'
    }

    #gets data
    response = requests.get(redirect_url, params=params)
    data = response.json()

    #clean data
    stock_data = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data = stock_data.astype(float)
    stock_data.rename(columns={'1. open' : 'open'}, inplace = True)

    #return minute to minute prices
    return stock_data['open'].sort_index(axis=0).to_frame(name = 'price')

#determines if stock moves pct(movement) up or down
def sell_at(start, stock_data : pd.DataFrame, close_situation : str, close_at_end_day : bool):
    #if you want to close at end day get only prices on this date
    if close_at_end_day:
        price = stock_data.loc[stock_data.index.date == start.date()]

    #gets stock data after datetime
    price = (price.loc[(price.index >= start)])['price']
    
    #gets stock data at possible sell points
    movement = price.to_frame(name='price').query(close_situation)['price']
    
    #returns pct change of trade if there is one
    if not movement.empty():
        return (movement.iloc[0] - price.iloc[0]) / price.iloc[0]
    
    #returns none if err
    return np.NaN

#tests success rate of each "situation"
def test_situation(situation, test_data : pd.DataFrame, stock_data : pd.DataFrame):
    #gets the datetime occurences of each "situation" 
    occurences = pd.Series(test_data.query(situation['open_situation']).index)

    #changes the trades to the direction of the indicate
    trades = occurences.apply(sell_at, args=(stock_data, situation['close_situation'], situation['close_at_end_of_day'])).dropna()

    return pd.Series({'test_occurences': len(trades), 'test_accuracy':  len(trades[trades==situation['indicate']]) / len(trades)})

#determines if the situation indicates to buy/sell/dont know
def determine_indication(train_data : pd.DataFrame, stock_data: pd.DataFrame, situation : str, close_situation : str, close_at_end_day : bool):
    #gets the datetime occurences of each "situation"
    occurences = pd.Series(train_data.query(situation).index)

    #performs "trades"
    trades = occurences.apply(sell_at, args=(stock_data, close_situation, close_at_end_day)).dropna()

    #calculates confidence interval
    confidence_interval = stats.t.interval(0.99, len(trades)-1, loc=np.mean(trades), scale=np.std(trades)/np.sqrt(len(trades)))
    
    #determines if both values in the confidence interval are the same sign(aka does this have an acutal indication)
    sign = (np.sign(confidence_interval[0]) + np.sign(confidence_interval[1])) / 2

    #returns a dict(gonna be appended to a df)
    if len(trades) != 0:
        return {'open_situation' : situation,
                'close_situation' : close_situation,
                'close_at_end_of_day' : close_at_end_day,
                'indicate' : sign,
                'interval' : confidence_interval,
                'train_occurences' : len(trades),
                'train_accuracy' : max(len(trades[trades>0]) / len(trades), len(trades[trades<0]) / len(trades))}
    
    return {'open_situation' : situation, 
            'close_situation' : close_situation,
            'close_at_end_of_day' : close_at_end_day,
            'indicate' : 0,
            'interval' : [0,0],
            'train_occurences' : 0,
            'train_accuracy' : 0}

def trade(test_data : pd.DataFrame, stock_data : pd.DataFrame, situations : pd.DataFrame):
    #determines success rate of each "situation" on test_data
    if not situations.empty:
        situations[['test_occurences', 'test_accuracy']] = situations.apply(test_situation, args=(test_data, stock_data), axis=1)

        situations['growth'] = (situations['test_occurences'] * (situations['test_accuracy'] - (1-situations['test_accuracy'])))
        
        #outputs that jawn
        return situations
    return 'u got no data'


if __name__ == "__main__":
    #kinda vars
    symbol = 'SPY'
    min_time = '09:30'
    max_time = '16:00'
    movement = 0.0025
    test_to_train_ratio = .5

    #get data from .env file
    load_dotenv()
    api_key = os.getenv("alpha_api_key")
    redirect_url = os.getenv('alpha_redirect_url')

    #initation
    situations = pd.DataFrame(columns=['situation', 'indicate', 'train_occurences', 'train_accuracy'])

    # gets stock data
    # stock_data_1 = get_data(symbol, api_key, redirect_url)
    stock_data = pd.read_pickle('alphavantage.pkl')
    # pd.to_pickle(pd.concat([stock_data, stock_data_1], axis = 0), 'alphavantage.pkl')
    
    #gets data during market open not including extra hours(which we can include later)
    stock_data = stock_data.between_time(min_time, max_time)

 
    # gets unique dates from stock_data and shuffles them
    # index_shuffled = np.array(stock_data.index)
    
    #shuffle split technique
    # np.random.shuffle(index_shuffled)
    # train_data = stock_data.loc[pd.to_datetime(stock_data.index).isin(index_shuffled[:int(len(index_shuffled) * test_to_train_ratio)])]
    # test_data = stock_data.loc[pd.to_datetime(stock_data.index).isin(index_shuffled[int(len(index_shuffled) * test_to_train_ratio):])]

    # # month splittage
    train_data = stock_data[(stock_data.index.month == 1)]
    test_data = stock_data[(stock_data.index.month == 2)]
  
    # tests these indicators and appends that indication to df
    # first string is your buy query, second string is your sell query
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(60) >= 1.0025', '(price >= price.iloc[0] * (1 + 0.0025)) | (price <= price.iloc[0] * (1 - 0.0025))', True), ignore_index = True).drop('situation', axis=1)

    #filters situations
    situations = situations.loc[(situations['indicate'] != 0)]

    print(situations)

    #performs trades
    trades = trade(test_data, stock_data, situations)

    print(trades)