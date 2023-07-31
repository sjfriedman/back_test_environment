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
def direction(start, stock_data : pd.DataFrame, movement = float):
    #gets stock data after date
    trade_data = (stock_data.loc[(stock_data.index.date == start.date()) & (stock_data.index >= start)])['price']
    
    #gets stock data of after it moves int(movement) percent up or down
    movement = trade_data.loc[(trade_data >= (trade_data.iloc[0] * (1 + movement))) | (trade_data <= (trade_data.iloc[0] * (1 - movement)))]

    try: 
        #returns sign if there is "movement"(.iloc[0] throws error when no data) within the same trading day
        return np.sign(movement.iloc[0] - trade_data.iloc[0])
    except:
        #returns None indicating market closed before the stock "moved"
        return np.NaN

#tests success rate of each "situation"
def test_situation(situation, test_data : pd.DataFrame, stock_data : pd.DataFrame, movement : float):
    #gets the datetime occurences of each "situation" 
    occurences = pd.Series(test_data.query(situation['situation']).index)

    #changes the trades to the direction of the indicate
    trades = occurences.apply(direction, args=(stock_data,movement)).dropna()

    return pd.Series({'test_occurences': len(trades), 'test_accuracy':  len(trades[trades==situation['indicate']]) / len(trades)})

#determines if the situation indicates to buy/sell/dont know
def determine_indication(train_data : pd.DataFrame, stock_data: pd.DataFrame, situation : str, movement : float):
    #gets the datetime occurences of each "situation"
    occurences = pd.Series(train_data.query(situation).index)

    #performs "trades"
    trades = occurences.apply(direction, args=(stock_data,movement,)).dropna()

    #calculates confidence interval
    confidence_interval = stats.t.interval(0.99, len(trades)-1, loc=np.mean(trades), scale=np.std(trades)/np.sqrt(len(trades)))
    
    #determines if both values in the confidence interval are the same sign(aka does this have an acutal indication)
    sign = (np.sign(confidence_interval[0]) + np.sign(confidence_interval[1])) / 2

    #returns a dict(gonna be appended to a df)
    if len(trades) != 0:
        return {'situation' : situation, 
            'indicate' : sign,
            'interval' : confidence_interval,
            'train_occurences' : len(trades),
            'train_accuracy' : max(len(trades[trades>0]) / len(trades), len(trades[trades<0]) / len(trades))}
    
    return {'situation' : situation, 
            'indicate' : 0,
            'interval' : [0,0],
            'train_occurences' : 0,
            'train_accuracy' : 0}

def trade(test_data : pd.DataFrame, stock_data : pd.DataFrame, situations : pd.DataFrame, movement : float):
    #determines success rate of each "situation" on test_data
    try: situations[['test_occurences', 'test_accuracy']] = situations.apply(test_situation, args=(test_data, stock_data, movement,), axis=1)
    except: return 'u got no data'

    situations['growth'] = (situations['test_occurences'] * (situations['test_accuracy'] - (1-situations['test_accuracy'])))

    #outputs that jawn
    return situations


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
    index_shuffled = np.array(stock_data.index)
    
    #shuffle split technique
    np.random.shuffle(index_shuffled)
    # train_data = stock_data.loc[pd.to_datetime(stock_data.index).isin(index_shuffled[:int(len(index_shuffled) * test_to_train_ratio)])]
    # test_data = stock_data.loc[pd.to_datetime(stock_data.index).isin(index_shuffled[int(len(index_shuffled) * test_to_train_ratio):])]

    # # month splittage
    train_data = stock_data[(stock_data.index.month <= 6)]
    test_data = stock_data[(stock_data.index.month > 6) & (stock_data.index.month <= 8)]
  
    # tests these indicators and appends that indication to df
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(60) >= 1.0025', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(60) <= 0.9975', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(60) >= 1.005', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(60) <= 0.995', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120) >= 1.005', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120) <= 0.995', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120) <= 0.995 & price / price.shift(60) <= .995', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120)  >= 1.005 & price / price.shift(60)  >= 1.005', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120) >= 0.995 & price / price.shift(60) <= .995', movement), ignore_index = True)
    situations = situations.append(determine_indication(train_data, stock_data, 'price / price.shift(120)  <= 1.005 & price / price.shift(60)  >= 1.005', movement), ignore_index = True)

    #filters situations
    situations = situations.loc[(situations['indicate'] != 0)]

    print(situations)

    #performs trades
    trades = trade(test_data, stock_data, situations, movement)


    print(trades)