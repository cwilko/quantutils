import numpy as np
import pandas as pd
from scipy import stats
import quantutils.core.timeseries as tms
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller


def sharpe(x):
    return (x.mean() / x.std()) * np.sqrt(252)


def getStats(x):
    return [x.mean(), x.var(), sharpe(x)]

# Stats on returns ts


def statistics(ts):
    m = ts.mean()
    s = ts.std()
    f = ts.mean() / (ts.std() ** 2)
    Sh = (m / s) * np.sqrt(252)
    maxDD, maxDDD = tms.maxDD(np.cumprod(1 + ts))

    print()
    print("## RESULTS ##")
    print()
    print("Sharpe Ratio (annualised) : {}".format(Sh))
    print("Mean Return (annualised) : {}%".format(m * 252 * 100))
    print("Standard Deviation (annualised) : {}%".format(s * np.sqrt(252) * 100))
    print("Maximum Drawdown (f=1) : {}%".format(maxDD * 100))
    print("Maximum Drawdown Duration (f=1) : {} periods".format(maxDDD))
    print("Max Return : {}%".format(max(ts) * 100))
    print("Min Return : {}%".format(min(ts) * 100))
    print()
    print("Optimal Kelly Stake : {}".format(f))


def statistical_tests(ts, sim, level):
    # SIGNIFICANCE TESTS
    MCMC(sim, ts)
    testRandom(sim, ts, level)
    confIntervals(ts, level)


# MONTE CARLO SIMULATION (BOOTSTRAP)
def bootstrap(ts, iterations=1000, txCost=0):
    print()
    print("Bootstrapping...")
    df = pd.DataFrame([getStats(ts.assign(result=lambda x: random_selection(x, txCost))["result"]) for i in range(0, iterations)], columns=['Mean', 'Var', 'Sharpe'])
    print("Completed {} iterations".format(iterations))
    print("Simulated Population Mean = {}".format(df['Mean'].mean()))
    print("Simulated Population Variance = {}".format(df['Var'].mean()))

    return df


def random_selection(x, txCost):
    direction = np.array([random.choice([-1, 1]) for i in range(len(x))])
    returns = direction * ((x["Close"] - x["Open"] - txCost) / x["Open"])
    return returns

# TEST SIGNIFICANTLY DIFFERENT SHARPE RATIO FROM RANDOM SELECTION


def MCMC(sim, ts):
    strat_sh = sharpe(ts)
    print()
    print("MCMC")
    # TODO: Shouldn't "sum" be "len"?
    print("Percent of Population greater than Strategy Sharpe {}%".format(100 * len(sim['Sharpe'][sim['Sharpe'] > strat_sh]) / len(sim)))
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

    print()
    print("Test for Random Distribution...")
    print()
    print("H0 : Strategy Returns = Random")
    print("H1 : Strategy Returns > Random")
    print()
    print("Random Population mean : {}".format(sim_m))
    print("Random Population stdev : {}".format(sim_s))
    print("Test Statistic (T-Score) : {}".format((strat_m - sim_m) / (strat_s / np.sqrt(n))))
    print("Critical Region : {}".format(stats.t.ppf(level, n - 1)))
    if ((strat_m - sim_m) / (strat_s / np.sqrt(n))) > stats.t.ppf(level, n - 1):
        print("H0 rejected at {}% confidence level".format(level * 100))
    else:
        print("H0 not rejected at {}% confidence level".format(level * 100))


def confIntervals(ts, level=.95):
    # CONFIDENCE INTERVALS

    n = len(ts)
    strat_m = ts.mean()
    strat_s = ts.std()
    t_score = stats.t.ppf(((1 + level) / 2), n - 1) * strat_s / np.sqrt(n)
    n_score = stats.norm.ppf(((1 + level) / 2)) * strat_s / np.sqrt(n)

    print()
    print("CONFIDENCE LEVELS : {}%".format(level * 100))
    print("Mean Return : {}".format(strat_m))
    print("Mean Return Confidence Intervals (T Dist): {}".format(strat_m - t_score, strat_m + t_score))
    print("Mean Return Confidence Intervals (Norm Dist): {}".format(strat_m - n_score, strat_m + n_score))
    print()

    # chi_score_L = ((n-1)*(strat_s**2)) / stats.chi2.ppf((1+lev`````````el)/2, n-1)
    chi_score_U = ((n - 1) * (strat_s**2)) / stats.chi2.ppf((1 - level), n - 1)

    print("Standard Deviation : {}".format(strat_s))
    print("StdDev Upper Confidence Level {}".format(np.sqrt(chi_score_U)))


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat))  # or percent error = * 100


def ARIMA(ts, pnl, predictStart, predictEnd):
    for AR in range(0, 4):
        for MA in range(0, 4):
            for I in range(0, 2):
                try:
                    arima_model = sm.tsa.ARIMA(ts, (AR, I, MA)).fit(trend='nc')
                    predict = arima_model.predict(predictStart, predictEnd, dynamic=False)
                    series = ts if I == 0 else pnl
                    print("ARIMA:{}{}{}, AIC:{}, MFE:{}, MAE:{}".format(AR, I, MA, arima_model.aic, mean_forecast_err(series, predict), mean_absolute_err(series, predict)))
                except:
                    print("Failed to fit {}{}{}".format(AR, I, MA))


def merton(model_ret, baseline_ret, display=False):

    theta = model_ret.values
    Z = baseline_ret.loc[model_ret.index].values

    N1 = sum(Z > 0)
    N2 = sum(Z <= 0)
    n1 = sum((Z > 0) & (theta > 0))
    n2 = sum((Z < 0) & (theta < 0))
    n = n1 + n2
    N = N1 + N2
    p1 = n1 / float(N1)
    p2 = (N2 - n2) / float(N2)
    p = p1 + p2
    mu = (n * N1) / float(N)
    sigma = (n1 * N1 * N2 * (N - n)) / float((N ** 2) * (N - 1))
    p_value = norm.sf(n1, loc=mu, scale=np.sqrt(sigma))

    if (display):
        print("Merton measure of Market Timing : " + str(round((1 - p_value) * 100, 2)) + "% predictability")
        print("p1 + p2 : " + str(p))
        print("p-value : " + str(p_value))
        print("")

    return p, p_value


def adf_test(x):
    return adfuller(x)
