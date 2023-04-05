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
    print("Optimal Kelly Stake : {}".format(f))


# Note:
# Only works for barOnly = True (period data)
# Annualisation only works if provided data is daily
# Only works for strategies where all periods are traded. E.g. cropTime or similar will reduce
# the mean/var artificially with 0's - even though these were "no trade" periods


def statistical_tests(ts, sim, level):
    # SIGNIFICANCE TESTS
    testRandom(sim, ts, level)
    confIntervals(ts, level)


# MONTE CARLO SIMULATION (BOOTSTRAP)
def bootstrap(ts, iterations=1000):
    print()
    print("Bootstrapping...")
    df = pd.DataFrame(
        [getStats(random_selection(ts)) for i in range(0, iterations)],
        columns=["Mean", "Var", "Sharpe"],
    )
    print("Completed {} iterations".format(iterations))
    print("Simulated Population Mean (Annualised) = {}".format(df["Mean"].mean() * 252))
    print(
        "Simulated Population StdDev (Annualised) = {}".format(
            np.sqrt(df["Var"].mean() * 252)
        )
    )
    print("Simulated Population Sharpe (Annualised) = {}".format(df["Sharpe"].mean()))

    return df


def random_selection(returns):
    return np.random.choice([-1, 1], returns.shape) * returns


def testRandom(sim, ts, level):
    # TEST SIGNIFICANTLY DIFFERENT FROM RANDOM POPULATION
    # H0 : Strategy Returns = Random Returns
    # H1 : Strategy Returns > Random Returns

    n = len(ts)
    strat_m = ts.mean() * 252
    strat_s = ts.std() * np.sqrt(252)

    sim_m = sim["Mean"].mean()
    sim_s = np.sqrt(sim["Var"].mean())
    SD = np.sqrt(((ts.var() * 252) + sim["Var"].mean()) / 2.0)
    t_score = (strat_m - sim_m) / (SD * np.sqrt(2.0 / n))

    print()
    print("Test for Random Distribution...")
    print()
    print("H0 : Strategy Returns = Random")
    print("H1 : Strategy Returns > Random")
    print()
    print("Random Population mean : {}".format(sim_m))
    print("Random Population stdev : {}".format(sim_s))
    print(
        "Percent of Population greater than Strategy Sharpe {}%".format(
            100 * len(sim["Sharpe"][sim["Sharpe"] > sharpe(ts)]) / len(sim)
        )
    )
    # print("Test Statistic (T-Score) : {}".format(t_score))
    # print("Critical Region : {}".format(stats.t.ppf(level, n - 1)))
    # if t_score > stats.t.ppf(level, n - 1):
    #    print("H0 rejected at {}% confidence level".format(level * 100))
    # else:
    #    print("H0 not rejected at {}% confidence level".format(level * 100))

    # Perform a 1-sample test of the mean vs the population mean (via simulation), to determine
    # if the sample mean is significantly different to the population. I.e. to see if
    # the strategy has not just been lucky.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    result = stats.ttest_1samp(ts, sim["Mean"].mean() / 252)
    print()
    # print("SCIPY RESULT:")
    print("Test Statistic (T-Score) : {}".format(result.statistic))
    print("Critical Value (P-Value) : {}".format(result.pvalue))
    print("Degrees of Freedom : {}".format(result.df))
    # print(str(result.confidence_interval(confidence_level=level)))
    if result.pvalue < (1 - level):
        print("H0 rejected at {}% confidence level".format(level * 100))
    else:
        print("H0 not rejected at {}% confidence level".format(level * 100))


def confIntervals(ts, level=0.95):
    # CONFIDENCE INTERVALS

    n = len(ts)
    strat_m = ts.mean()
    strat_s = ts.std()
    t_score = stats.t.ppf(((1 + level) / 2), n - 1) * strat_s / np.sqrt(n)
    n_score = stats.norm.ppf(((1 + level) / 2)) * strat_s / np.sqrt(n)

    print()
    print("CONFIDENCE LEVELS : {}%".format(level * 100))
    print("Mean Return : {}".format(strat_m))
    print(
        "Mean Return Confidence Intervals (T Dist): {}, {}".format(
            strat_m - t_score, strat_m + t_score
        )
    )
    print(
        "Mean Return Confidence Intervals (Norm Dist): {}, {}".format(
            strat_m - n_score, strat_m + n_score
        )
    )
    print()

    # chi_score_L = ((n-1)*(strat_s**2)) / stats.chi2.ppf((1+lev`````````el)/2, n-1)
    chi_score_U = ((n - 1) * (strat_s**2)) / stats.chi2.ppf((1 - level), n - 1)

    print("Standard Deviation : {}".format(strat_s))
    print("StdDev Upper Confidence Level {}".format(np.sqrt(chi_score_U)))


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat))  # or percent error = * 100


def ARIMAFit(ts, order=None, display=True):

    if not order:  # Search orders
        bestAIC = np.inf
        for AR in range(0, 5):
            for MA in range(0, 5):
                for I in range(0, 2):
                    tmpResult = ARIMAFit(ts, order=(AR, I, MA), display=display)

                    if tmpResult.aic < bestAIC:
                        bestAIC = tmpResult.aic
                        result = tmpResult
                        best_order = (AR, I, MA)
        if display:
            print()
            print(f"Best Result was ARIMA{best_order} with an AIC of {bestAIC}")
    else:
        try:
            result = sm.tsa.arima.ARIMA(ts, order=order).fit()
            predict = result.predict(start=0, end=len(ts) - 1, dynamic=False)

            if display:
                print(
                    "ARIMA:{},{},{}, AIC:{}, MFE:{}, MAE:{}".format(
                        order[0],
                        order[1],
                        order[2],
                        result.aic,
                        mean_forecast_err(ts, predict),
                        mean_absolute_err(ts, predict),
                    )
                )
        except Exception as e:
            print(f"Failed to fit ({order[0]}, {order[0]}, {order[0]})")
            # print(e)

    return result


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
    sigma = (n1 * N1 * N2 * (N - n)) / float((N**2) * (N - 1))
    p_value = norm.sf(n1, loc=mu, scale=np.sqrt(sigma))

    if display:
        print(
            "Merton measure of Market Timing : "
            + str(round((1 - p_value) * 100, 2))
            + "% predictability"
        )
        print("p1 + p2 : " + str(p))
        print("p-value : " + str(p_value))
        print("")

    return p, p_value


def adf_test(x, opts):
    return adfuller(x, **opts)
