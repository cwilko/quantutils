from numpy import *
from finance import *
from uuid import *

def MA(values, period):
    ma = [0]*len(values)    
    for n in range(period-1,len(ma)):
        ma[n] = mean(values[n-(period-1):n])
    ma[0:period] = [ma[period-1]]*period
    return ma
    
def MA_prog(values, period):
    ma = [0]*len(values)
    for n in range(0, len(ma)):
        if n < period :
            ma[n] = mean(values[0:n+1])
        else:
            ma[n] = mean(values[n-(period-1):n+1])
    return ma
    
def DMA(values, period, offset):
    ma = [0]*len(values)    
    for n in range(period-1,len(ma)-offset):
        ma[n+offset] = mean(values[n-(period-1):n])
    ma[0:period+offset] = [ma[period-1+offset]]*(period+offset)
    return ma

def DMA_prog(values, period, offset):
    ma = MA_prog(values, period) 
    dma = [0]*len(values) 
    if offset > 0:   
        for n in range(offset - 1,len(ma)):
            #if n < (offset-1):
            #    dma[n] = DMA_prog(values, 2*(n+1), n+1)[n]
            #else:
                dma[n] = ma[n - (offset - 1)]
    elif offset < 0:
        for n in range(0, len(ma) - abs(offset)):
            dma[n] = ma[n + (abs(offset) - 1)]
    else:
        dma = ma
    return dma
    
def calculateInflexions(period, prices, dates, data_start):
    
    # Algorithm variables
    sample_count = data_start
    PTSample = 0
    PTDate = 0
    sample = 0

    dt = dtype({'names':('price','date'),'formats':('float','|O4')})
    inflexions = array([],dt)
    
    # Algorithm
    
    dma_lag = MA_prog(prices, period)
    dma_lead = DMA_prog(prices, period, period/2)
    
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
    
    highwatermark=zeros(len(cumret)); # initialize high watermarks to zero.

    drawdown=ones(len(cumret)); # initialize drawdowns to zero.

    drawdownduration=zeros(len(cumret)); # initialize drawdown duration to zero.

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
