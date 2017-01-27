import random
import pandas as pd
from tradeFramework import TradeEngine

def gap_close(x, txCost):
    profit = 0.0
    ohlc = x.iloc[0]
    print x
    
    if (gap_close_predict(x)["bar"] == TradeEngine.BUY ):
        #if (ohlc.Low < ohlc.prevClose):
        #    profit += (ohlc.Open - ohlc.prevClose)
        #else:
        profit += (ohlc.Close - ohlc.Open - txCost) / ohlc.Open
    else:
        #if (ohlc.High > ohlc.prevClose):
        #    profit += (ohlc.prevClose - ohlc.Open)
        #else:
        profit += (ohlc.Open - ohlc.Close - txCost) / ohlc.Open
    return profit

def gap_close_predict(x):
    ohlc = x.iloc[0]
    if (ohlc.Open > ohlc.prevClose):
        return pd.Series({'bar':TradeEngine.BUY, 'gap':TradeEngine.CASH})
    else:
        return pd.Series({'bar':TradeEngine.SELL, 'gap':TradeEngine.CASH})

def gap_close_prices(x, txCost):
    profit = 0.0
    ohlc = x.iloc[0]
    
    if (ohlc.Open > ohlc.prevClose):
        #if (ohlc.Low < ohlc.prevClose):
        #    profit += (ohlc.Open - ohlc.prevClose)
        #else:
        profit += (ohlc.Close - ohlc.Open - txCost)
    else:
        #if (ohlc.High > ohlc.prevClose):
        #    profit += (ohlc.prevClose - ohlc.Open)
        #else:
        profit += (ohlc.Open - ohlc.Close - txCost)
    return profit

def random_selection(x, txCost):
    profit = 0.0
    ohlc = x.iloc[0]    
    buy = random.choice([True, False])
    if (buy):
        #if (ohlc.Low < ohlc.prevClose):
        #    profit += (ohlc.Open - ohlc.prevClose)
        #else:
        profit += (ohlc.Close - ohlc.Open - txCost) / ohlc.Open
    else:
        #if (ohlc.High > ohlc.prevClose):
        #    profit += (ohlc.prevClose - ohlc.Open)
        #else:
        profit += (ohlc.Open - ohlc.Close - txCost) / ohlc.Open
    return profit

def buy_and_hold(x, txCost):
    proft = 0.0
    ohlc = x.iloc[0]  
    profit += (ohlc.Close - ohlc.Open - txCost) / ohlc.Open
    return profit