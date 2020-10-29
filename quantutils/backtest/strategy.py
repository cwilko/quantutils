import numpy as np
from uuid import *

# ======================
# Strategy Class
# ======================


class Strategy:

    def __init__(self, net_return, name, period, txCost):
        ''' Unleveraged Returns & Statistics '''
        self.name = name
        self.period = period
        self.net_return = net_return
        self.mean = mean(self.net_return)
        self.var = var(self.net_return)
        self.std = std(self.net_return)
        self.amean = self.mean * period
        self.kelly = self.mean / self.var
        self.txCost = txCost

    def display(self):

        # Strategy Statistics
        print()
        print("=" * (len(self.name) + 14))
        print("Strategy ID : %s " % self.name)
        print("Aritmetic Mean = %.2f%%" % (self.amean * 100))
        print("Volatility = %.2f%%" % (self.std * sqrt(self.period) * 100))
        print("Sharpe Ratio = %.2f" % self.sharpe)
        print("Kelly Optimal Leverage = %.2f" % self.kelly)

        # Execution Statistics
        print("Strategy Leverage = %.2f" % self.leverage)
        print("Geometric Mean Annual Return = %.2f%%" % (self.gmean * 100))
        print("Max Drawdown = %.2f%%" % (self.maxDD * 100))
        print("Max Drawdown Duration = %d days" % self.maxDDD)
        print("Return over Maximum Drawdown (RoMaD) = %.2f" % self.romad)
        print("APR = %.2f%%" % (self.APR * 100))
        print("=" * (len(self.name) + 14))


class StocksStrategy(Strategy):

    def __init__(self, net_return, name=uuid4().hex, period=252.0, txCost=0.0):
        Strategy.__init__(self, net_return, name, period, txCost)

        '''
        if self.kelly > 1:
            rate = debitRate
        else:
            rate = creditRate

        self.kelly = (self.mean - (rate / period)) / self.var
        self.sharpe = sqrt(period) * (self.mean - (rfRate / period)) / self.std

        self.leverage = 1
        if kelly>0:
            self.leverage = kelly * self.kelly

        compound_return = cumprod(1 + (self.leverage * self.net_return + (1.0 - self.leverage) * ((1 + rate) ** (1.0/self.period) - 1)))
        self.compound_return = np.insert(compound_return,0,1)

        # Difference between gmean and APR due to daily compounded returns (APR) rather than continuously compounded returns (gmean)
        self.gmean = exp(rate + self.leverage * ((self.mean * self.period) - rate) - ((self.var * self.period * (self.leverage ** 2)) / 2.0))-1
        self.APR = self.compound_return[-1] ** (1.0 * self.period/float(len(self.compound_return)-1)) - 1 # Nth root of the final capital
        self.APR = exp(log(self.compound_return[-1]) / ((len(self.compound_return)-1) / self.period)) - 1 # exp(log of final capital / T)

        self.maxDD, self.maxDDD = maxDD(self.compound_return)
        self.romad = self.APR / self.maxDD
        '''


class SpreadsStrategy(Strategy):

    def __init__(self, net_return, prices, name=uuid4().hex, period=252.0, txCost=0.0):
        Strategy.__init__(self, net_return, name, period, txCost)

        self.prices = prices
        '''
        # Kelly is (mPn - r)/(s2Pn2), or f1 - r/s2Pn2
        # However the r component approaches 0 so we consider r=0 for display, and use the actual values for calculating g
        self.kelly = self.mean / self.var

        kelly_values = (self.mean / (self.prices * self.var))
        rate = zeros(len(kelly_values))
        for i, value in enumerate(kelly_values):
            if value > 1.0:
                rate[i] = debitRate
            else:
                rate[i] = creditRate

        kelly_values = (((self.prices * self.mean) - (rate/self.period)) / ((self.prices ** 2) * self.var))

        self.leverages = self.leverage = 1.0
        if kelly>0:
            self.leverage = kelly * self.kelly
            self.leverages = kelly * kelly_values


        # Sharpe is actually, (mPn - r)/(sPn), or S - r/sPn
        # However the r component approaches 0, so we just consider r=0
        # TODO : Sharpe also depends on txCosts, i.e. (mPn - r - a|X|)/sPn, where X
        self.sharpe = sqrt(self.period) * (self.mean / self.std)
        '''
        ''' Geometric Mean '''
        '''
        # g = r + f(m-r) + f^2s^2/2
        # Note gmean > APR due to daily compounded returns (APR) rather than continuously compounded returns (gmean)

        #self.g = (self.mean ** 2) / ((self.var * 2.0)) # Only applicable if r=0, f=f*
        self.g = (rate / self.period) + self.leverages * ((self.mean * self.prices) - (rate / self.period)) - ((self.var * (self.leverages ** 2) * (self.prices ** 2)) / 2.0)
        self.gmean = exp((sum(self.g) / (len(self.g)) * self.period))-1
        '''
        ''' Calculate the actual compound returns '''
        '''
        compound_return = cumprod(1 + (self.leverages * (self.prices * self.net_return) + (1.0 - self.leverages) * ((1 + creditRate) ** (1.0/self.period) - 1)))
        self.compound_return = np.insert(compound_return,0,1)

        self.APR = self.compound_return[-1] ** (1.0 * self.period/float(len(self.compound_return)-1)) - 1 # Nth root of the final capital
        self.APR = exp((log(self.compound_return[-1]) / (len(self.compound_return)-1)) * self.period) - 1 # exp(log of final capital / T)

        self.maxDD, self.maxDDD = maxDD(self.compound_return)
        self.romad = self.APR / self.maxDD
        '''


# ======================
# Portfolio Class
# ======================

class Portfolio(Strategy):

    def __init__(self, strategyList, kelly=0.0, debitRate=0.0, creditRate=0.0, name=uuid4().hex, period=252.0, rfRate=0.04):

        self.name = name
        self.period = period
        self.strategyList = strategyList

        n = len(strategyList)

        ''' Calculate relative proportions of each strategy based on kelly criterion '''

        # Calculate the covariance matrix, C, and the mean vector, M
        returns = empty(n, dtype=object)
        means = empty(n)
        alpha = empty(n)
        for i, strategy in enumerate(strategyList):
            returns[i] = strategy.net_return
            means[i] = strategy.mean
            alpha[i] = strategy.txCost
        self.returns = returns

        M = matrix(means).T
        C = matrix(cov(vstack(returns)))

        # Calculate the optimal kelly vector, F
        F = (C.I * M)
        self.kelly = sum(F)
        if kelly == 0.0:
            F = F / sum(F)
        else:
            F = F * kelly

        if sum(abs(F)) > 1:
            rate = debitRate
            F = (C.I * (M - (rate / period)))
            self.kelly = sum(F)
            F = F * kelly
        else:
            rate = creditRate

        self.M = M
        self.C = C
        self.F = F
        self.leverage = sum(F)

        ''' Calculate the net return of the portfolio '''

        # TODO : Rebalance amount is not correct formula for +/- returns (should be 1-f on denominator if r<0)
        self.net_return = 0
        for i, strategy in enumerate(strategyList):
            X = F.A1[i] * returns[i] * (1 - abs(F.A1[i])) / (1 - alpha[i] * abs(F.A1[i]))  # rebalance amount
            self.net_return = self.net_return \
                + (F.A1[i] * returns[i]) \
                + (1.0 - sum(abs(F))) * ((1 + rate) ** (1.0 / self.period) - 1) \
                - alpha[i] * abs(X)

        ''' Calculate the mean, gmean, and std '''

        #self.mean = (rate / period) + (F.T * (M - rate / period))
        #self.var = F.T * C * F

        self.mean = mean(self.net_return)
        self.var = var(self.net_return)
        self.std = sqrt(self.var)
        self.gmean = exp((self.mean - (self.var / 2)) * period) - 1
        self.amean = self.mean * period
        self.sharpe = sqrt(period) * (self.mean - (rfRate / period)) / self.std

        ''' Calculate the compound return of the portfolio '''

        self.compound_return = cumprod(1.0 + self.net_return)
        self.compound_return = np.insert(self.compound_return, 0, 1)

        self.APR = exp(log(self.compound_return[-1]) / ((len(self.compound_return) - 1) / self.period)) - 1  # exp(log of final capital / T)

        self.maxDD, self.maxDDD = maxDD(self.compound_return)
        self.romad = self.APR / self.maxDD

# ======================
# Spreads Portfolio Class
# ======================


class SpreadsPortfolio(Strategy):

    def __init__(self, strategyList, kelly=0.0, debitRate=0.0, creditRate=0.0, name=uuid4().hex, period=252.0, rfRate=0.04):

        self.name = name
        self.period = period
        self.strategyList = strategyList

        n = len(strategyList)

        ''' Calculate relative proportions of each strategy based on kelly criterion, Fi = (Qi*C).I * (M*Pi)'''

        # Calculate the covariance matrix, C, and the mean vector, M
        returns = [None] * n
        prices = empty(n, dtype=object)
        means = empty(n)
        alpha = empty(n)
        for i, strategy in enumerate(strategyList):
            returns[i] = strategy.net_return
            prices[i] = strategy.prices
            means[i] = strategy.mean
            alpha[i] = strategy.txCost
        self.returns = returns
        self.prices = vstack(prices)
        self.alpha = alpha

        M = matrix(means).T
        C = matrix(cov(returns))

        # Calculate the n matrices of covariance coefficients, Qi

        m = len(vstack(returns)[0])
        Q = empty([m], dtype=object)
        for i in range(0, m):
            q = empty([n, n])
            for j in range(0, n):
                q[j] = self.prices[:, i] * self.prices[:, i][j]
            Q[i] = matrix(q)

        # Calculate F
        F = empty([m], dtype=object)
        for i in range(0, m):
            F[i] = (multiply(Q[i], C).I * multiply(M, vstack(self.prices[:, i])))

        self.kelly = self.prices[:, 0] * F[0]

        for i in range(0, m):
            if kelly == 0.0:
                F[i] = F[i] / sum(F[i])
            else:
                F[i] = F[i] * kelly

        if sum(abs(F[0])) > 1:
            rate = debitRate
            F = empty([m], dtype=object)
            for i in range(0, m):
                F[i] = (multiply(Q[i], C).I * (multiply(M, vstack(self.prices[:, i])) - (rate / period)))

            self.kelly = self.prices[:, 0] * F[0]

            for i in range(0, m):
                if kelly == 0.0:
                    F[i] = F[i] / sum(F[i])
                else:
                    F[i] = F[i] * kelly
        else:
            rate = creditRate

        self.M = M
        self.C = C
        self.F = F
        self.Q = Q
        self.leverage = self.prices[:, 0] * F[0]

        ''' Calculate the net return of the portfolio '''

        self.gross_return = 0
        self.interest = (1 - array([sum(abs(f)) for f in F])) * ((1 + rate) ** (1.0 / self.period) - 1)
        self.txCosts = empty(n, dtype=object)
        Fstack = vstack([f.A1 for f in F])
        for i, strategy in enumerate(strategyList):
            self.txCosts[i] = (Fstack[:, i] * returns[i] * self.prices[i] * Fstack[:, i]) / (1 + alpha[i] * abs(Fstack[:, i]))  # rebalance amount
            self.gross_return = self.gross_return \
                + (Fstack[:, i] * returns[i] * self.prices[i]) \
                - alpha[i] * abs(self.txCosts[i])
        self.net_return = self.gross_return + self.interest

        ''' Calculate the mean, gmean, and std '''

        #self.mean2 = (rate / period) + ((multiply(M.T,self.prices[:,0]) - (rate / period)) * F[0])
        #self.var2 = F[0].T * multiply(Q[0],C) * F[0]

        self.mean = mean(self.net_return)
        self.var = var(self.net_return)
        self.std = sqrt(self.var)
        self.amean = self.mean * period
        self.gmean = exp((self.mean - (self.var / 2)) * period) - 1
        self.sharpe = sqrt(period) * (self.mean - (rfRate / period)) / self.std

        ''' Calculate the compound return of the portfolio '''

        self.compound_return = cumprod(1.0 + self.net_return)
        self.compound_return = insert(self.compound_return, 0, 1)

        self.APR = exp(log(self.compound_return[-1]) / ((len(self.compound_return) - 1) / self.period)) - 1  # exp(log of final capital / T)

        self.maxDD, self.maxDDD = maxDD(self.compound_return)
        self.romad = self.APR / self.maxDD
