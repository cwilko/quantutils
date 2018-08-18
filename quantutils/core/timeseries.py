import numpy as np

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_function(values, function, window, offset=0):
    x = function(rolling_window(values, window))
    pad = int(np.floor((window/2)-.5)+offset)
    r = np.empty(len(values)) * np.nan
    r[pad:pad+len(x)] = x
    return r

# Return values of an autocorrelation function
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    result = result / max(result)
    return result

def MA(values, window, offset=0):
    _fun = lambda x: np.mean(x, axis=1)
    return rolling_function(values, _fun, window, offset)

def EMA(values, window, offset=0):
    return np.roll(values.ewm(span=window, min_periods=window).mean(), -(window//2)+offset)

def MStd(values, window, offset=0):
    _fun = lambda x: np.std(x, axis=1)
    return rolling_function(values, _fun, window, offset)

def MVar(values, window, offset=0):
    _fun = lambda x: np.var(x, axis=1)
    return rolling_function(values, _fun, window, offset)

def MACF(values, lag, window, offset=0):
    _fun = lambda x: [autocorr(w)[lag] for w in x]
    return rolling_function(values, _fun, window, offset)
    
def calculateInflexions(period, prices, dates, data_start):
    
    # Algorithm variables
    sample_count = data_start
    PTSample = 0
    PTDate = 0
    sample = 0

    dt = np.dtype({'names':('price','date'),'formats':('float','|O4')})
    inflexions = np.array([],dt)
    
    # Algorithm
    
    dma_lag = MA(prices, period, period//2)
    dma_lead = MA(prices, period)
    
    uptrend = 0
    if dma_lag[sample_count] > dma_lead[sample_count]:
        uptrend = 1
    PTSample = prices[sample_count]
    PTDate = dates[sample_count]
    while sample_count < len(prices) - 1:
        
        sample = prices[sample_count]
        
        if uptrend:
            if (dma_lead[sample_count] <= dma_lag[sample_count]):
                if (sample > PTSample):
                    PTSample = sample
                    PTDate = dates[sample_count]
            else:                
                uptrend = 0
                inflexions.resize(len(inflexions)+1)
                inflexions[len(inflexions)-1] = (PTSample, PTDate)
                #import pdb; pdb.set_trace()
                PTSample = prices[sample_count+1]
                PTDate = dates[sample_count+1]
        else:
            if dma_lag[sample_count] < dma_lead[sample_count]:
                if (sample < PTSample):
                    PTSample = sample
                    PTDate = dates[sample_count]
            else:
                uptrend = 1
                inflexions.resize(len(inflexions)+1)
                inflexions[len(inflexions)-1] = (PTSample, PTDate)
                #import pdb; pdb.set_trace()
                PTSample = prices[sample_count+1]
                PTDate = dates[sample_count+1]
    
        sample_count += 1
    
    inflexions.resize(len(inflexions)+1)
    inflexions[len(inflexions)-1] = (PTSample, PTDate)
    
    return inflexions

def maxDD(cumret):
    
    highwatermark=np.zeros(len(cumret)); # initialize high watermarks to zero.

    drawdown=np.ones(len(cumret)); # initialize drawdowns to zero.

    drawdownduration=np.zeros(len(cumret)); # initialize drawdown duration to zero.

    for t in range(1,len(cumret)):
        highwatermark[t]=max(highwatermark[t-1], cumret[t]);
        drawdown[t]=(cumret[t]/highwatermark[t]); # drawdown on each day
        if (drawdown[t]==1):
            drawdownduration[t]=0;
        else:
            drawdownduration[t]=drawdownduration[t-1]+1;
    
    maxDD = 1 - min(drawdown); # maximum drawdown

    maxDDD=max(drawdownduration); # maximum drawdown duration
    
    return [maxDD, maxDDD]
