import numpy as np
import datetime as dt

STOCK_ROOT = '/home/cwilkin/Documents/Stock'
# Windows
# STOCK_ROOT = "E:/Stock"


def loadStockData(stockFile, startDate, endDate):
    dateconvert = lambda x: dt.datetime.strptime(x, "%m/%d/%Y").date()
    M = np.loadtxt(open(stockFile, "rb"), delimiter=",", skiprows=0, converters={0: dateconvert}, dtype={'names': ('Date', 'Open', 'High', 'Low', 'Close', 'Volume'), 'formats': (object, 'float', 'float', 'float', 'float', 'float')})

    startIndex = np.where(M['Date'] == dateconvert(startDate))[0][0]
    endIndex = np.where(M['Date'] == dateconvert(endDate))[0][0]
    return M[startIndex:endIndex]


def loadPair(stockFile1, stockFile2, startDate, endDate):
    num_data = loadStockData(stockFile1, startDate, endDate)
    num_data2 = loadStockData(stockFile2, startDate, endDate)

    close_prices = num_data['Close']
    dates = num_data['Date']

    close_prices2 = num_data2['Close']
    dates2 = num_data2['Date']

    return sync_pair_dates(dates, close_prices, dates2, close_prices2)


def loadStocks(stocks, startDate, endDate):
    M = [loadStockData(s, startDate, endDate) for s in stocks]
    return sync_dates(M)


def sync_pair_dates(dates1, data1, dates2, data2):
    N = min(len(dates1), len(dates2))
    result = np.zeros(N, dtype={'names': ('date', 'data1', 'data2'), 'formats': (object, 'float', 'float')})
    n = 0
    for i in range(0, N):
        for j in range(0, N):
            if dates1[i] == dates2[j]:
                result[n] = (dates1[i], data1[i], data2[j])
                n = n + 1
                break

    result = result[0:n]
    return result


def sync_dates(stocks):
    numstocks = len(stocks)
    days = len(stocks[0])
    results = np.vstack([np.empty(days, dtype={'names': ('Date', 'Open', 'High', 'Low', 'Close', 'Volume'), 'formats': (object, 'float', 'float', 'float', 'float', 'float')})] * numstocks)
    n = 0
    for i in range(0, days):
        results[0, n] = (stocks[0][i][0], stocks[0][i][1], stocks[0][i][2], stocks[0][i][3], stocks[0][i][4], stocks[0][i][5])
        commonDate = True
        for j in range(1, numstocks):
            for k in range(0, len(stocks[j])):
                if stocks[0]['Date'][i] == stocks[j]['Date'][k]:
                    results[j, n] = (stocks[j][k][0], stocks[j][k][1], stocks[j][k][2], stocks[j][k][3], stocks[j][k][4], stocks[j][k][5])
                    commonDate = True
                    break
                else:
                    commonDate = False
            if commonDate == False:
                break
        if commonDate == True:
            n = n + 1

    results = results[:, :n]
    return results
