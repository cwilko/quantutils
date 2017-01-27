import numpy as np
import pandas as pd
from scipy import stats
import strategies
import trading as td
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
import statsmodels.api as sm

def sharpe(x):
    return (x.mean() / x.std()) * np.sqrt(252)

def getStats(x):
    return [x.mean(), x.var(), sharpe(x)]

def statistics(ts):
    m = ts.mean()
    s = ts.std() 
    f = ts.mean() / (ts.std() ** 2)
    Sh = (m / s) * np.sqrt(252)
    maxDD, maxDDD = td.maxDD(np.cumprod(1 + ts))
    print
    print "## RESULTS ##"
    print
    print "Sharpe Ratio (annualised) : ", Sh
    print "Mean Return (annualised) : ", m * 252 * 100, "%"
    print "Standard Deviation (annualised) : ", s * np.sqrt(252) * 100, "%"
    print "Maximum Drawdown (f=1) : ", maxDD * 100, "%"
    print "Maximum Drawdown Duration (f=1) : ", maxDDD, "periods"
    print "Max Return : ", max(ts) * 100, "%"
    print "Min Return : ", min(ts) * 100, "%"
    print 
    print "Optimal Kelly Stake : ", f

def tests(sim, ts, level):
    # SIGNIFICANCE TESTS
    confIntervals(ts, level)
    MCMC(sim, ts)
    testRandom(sim, ts, level)
    
    
# MONTE CARLO SIMULATION (BOOTSTRAP)
def bootstrap(ts, iterations=1000, txCost=0, log=False):   
    print
    print "Bootstrapping..."
    df = pd.DataFrame([getStats(ts.groupby(pd.TimeGrouper(freq='B')).apply(strategies.random_selection, txCost)) for i in  range(0,iterations)], columns=['Mean','Var','Sharpe'])
    
    print "Completed ", iterations, " iterations"
    print "Simulated Population Mean = ", df['Mean'].mean()
    print "Simulated Population Variance = ", df['Var'].mean()
    
    return df

# TEST SIGNIFICANTLY DIFFERENT SHARPE RATIO FROM RANDOM SELECTION
def MCMC(sim, ts):   
    strat_sh = sharpe(ts)
    print
    print "MCMC"
    print "Percent of Population greater than Strategy Sharpe", 100*sum(sim['Sharpe'][sim['Sharpe']>strat_sh])/len(sim),"%"
    return
    
def testRandom(sim, ts, level):
    # TEST SIGNIFICANTLY DIFFERENT FROM RANDOM POPULATION
    # H0 : Strategy Returns = Random Returns
    # H1 : Strategy Returns > Random Returns

    n = len(ts)
    strat_m = ts.mean()
    strat_s = ts.std()

    sim_m = sim['Mean'].mean()
    sim_s = np.sqrt(sim['Var'].mean())

    print 
    print "H0 : Strategy Returns = Random"
    print "H1 : Strategy Returns > Random"
    print 
    print "Random Population mean :", sim_m
    print "Random Population stdev :", sim_s
    print "Test Statistic (T-Score) :", (strat_m - sim_m) / (strat_s / np.sqrt(n))
    print "Critical Region : ", stats.t.ppf(level, n-1)
    if ((strat_m - sim_m) / (strat_s / np.sqrt(n))) > stats.t.ppf(level, n-1):
        print "H0 rejected at ", level * 100, "% confidence level"
    else:
        print "H0 not rejected at ", level * 100, "% confidence level"
    return

def confIntervals(ts, level=.95):
    # CONFIDENCE INTERVALS

    n = len(ts)
    strat_m = ts.mean()
    strat_s = ts.std()
    t_score = stats.t.ppf(((1 + level) / 2) , n-1) * strat_s / np.sqrt(n)
    n_score = stats.norm.ppf(((1 + level) / 2)) * strat_s / np.sqrt(n)
    
    print
    print "CONFIDENCE LEVELS : ", level*100, "%"
    print "Mean Return :", strat_m 
    print "Mean Return Confidence Intervals (T Dist):", strat_m - t_score, strat_m + t_score
    print "Mean Return Confidence Intervals (Norm Dist):", strat_m - n_score, strat_m + n_score
    print

    #chi_score_L = ((n-1)*(strat_s**2)) / stats.chi2.ppf((1+lev`````````el)/2, n-1)
    chi_score_U = ((n-1)*(strat_s**2)) / stats.chi2.ppf((1-level), n-1)

    print "Standard Deviation :", strat_s
    print "StdDev Upper Confidence Level ", np.sqrt(chi_score_U)
    
def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

def ARIMA(ts, pnl, predictStart, predictEnd):
    for AR in range(0, 4):
        for MA in range(0, 4):
            for I in range(0,2):                
                try:
                    arima_model = sm.tsa.ARIMA(ts, (AR,I,MA)).fit(trend='nc')
                    predict = arima_model.predict(predictStart, predictEnd, dynamic=False)
                    series = ts if I == 0 else pnl
                    print "ARIMA:", AR, I, MA, ", AIC:", arima_model.aic, ", MFE:", mean_forecast_err(series, predict), ", MAE:", mean_absolute_err(series, predict)
                except:
                    print "Failed to fit ", AR, I, MA
    
def plot(ts, pnl, f=1):
    plt.figure(0)
    plt.title("Strategy Returns")
    plt.plot(np.cumsum(np.log(1 + (pnl * f))))
    plt.show()
    
    # DATA PLOT
    fig = plt.figure(1)
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    #ax.xaxis.set_minor_formatter(dayFormatter)

    #plot_day_summary(ax, quotes, ticksize=3)
    ts['d'] = ts.index.map(date2num)

    candlestick_ohlc(ax, ts[['d', 'Open','High', 'Low', 'Close']].astype('float32').values, width=0.6)
   
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title("Underlying Security Prices")
    plt.show()
   