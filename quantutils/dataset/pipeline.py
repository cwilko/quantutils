import pandas
import numpy

## 
## Merge data
##

def merge(newData, existingData):
    print "Merging data..."
    return existingData.combine_first(newData)

##
## Shuffle data
##
def shuffle(data):
    return data.sample(frac=1).reset_index(drop=True)
    
##
## Concatenate Columns
##

def concat(data1, data2, axis=1):
    print("Concatenating features %s with classifications %s" % (data1.shape, data2.shape))
    return pandas.DataFrame(numpy.concatenate([data1.values, data2.values], axis=axis), data1.index)

##
## Crop
##

def cropDate(data, start, end):
    return data[start:end]

def cropTime(data, start, end):
    return data.between_time(start, end, include_start=True, include_end=False)

##
## Resample
##

def resample(data, sample_unit):
    print("Resampling to %s periods" % sample_unit)
    order = data.columns
    return data.resample(sample_unit).agg({'Open': 'first', 'High': lambda x : x.max(skipna=False), 'Low': lambda x : x.min(skipna=False),'Close': 'last'})[order]

##
## Remove Missing Data (NaN)
##

def removeNaNs(data):
    return data.dropna()
##
## Convert to Feature Matrix
##

def toFeatureSet(data, feature_periods):
    n = data.values.shape[1] * feature_periods
    return pandas.DataFrame(reshape(data, n),data.index[range(0,len(data),feature_periods)])

def reshape(ts, n):
    return numpy.reshape(ts.values[0:ts.values.size / n * n / ts.values.shape[1]], (ts.values.size / n, n))

##
## Encode classification
##

def encode(data, encoding):
    nanIndex = data.isnull().any(axis=1)
    if (encoding == "binary"):
        df = pandas.DataFrame((data.values[:,-1] > data.values[:,0]).astype(float), data.index)
    if (encoding == "one-hot"):
        df = pandas.DataFrame(numpy.column_stack
                                ([(data.values[:,-1] > data.values[:,0]).astype(float), 
                                  (data.values[:,0] > data.values[:,-1]).astype(float)])
                                , data.index)

    df.values[nanIndex.values] = numpy.nan

    return df

##
## Convert to local time zone
##
import pytz
def localize(data, sourceTZ, targetTZ):
    print "Converting from " + sourceTZ + " to " + targetTZ
    timezone = pytz.timezone(targetTZ)
    data.index = data.index.tz_localize(sourceTZ, ambiguous='NaT').tz_convert(timezone)
    return data

##
## Normalise (Candlesticks)
##
def normaliseCandlesticks(data, allowZero=True):
    X = data.values
    Xmax = X.max(axis=1)[numpy.newaxis].T
    Xmin = X.min(axis=1)[numpy.newaxis].T
    scale = Xmax - Xmin
    X = (X - Xmin) / scale
    if (allowZero!=True):
        X = (X * 2) - 1
    return pandas.DataFrame(numpy.hstack((X,scale / numpy.nanmax(scale))), data.index)

##
## Split (Train/Val/Test)
##
def split(data, train=.6, val=.2, test=.2):
    idx = numpy.arange(0,len(data)) / float(len(data))
    msk1 = data[idx<train]
    msk2 = data[(idx>=train) & (idx<(train + val))]
    msk3 = data[(idx>=(train+val))]
    return [msk1, msk2, msk3]

## Split with absolute value (of test set)
def splitAbs(data, testSetLength):
    return [data[:-(testSetLength)], data[-(testSetLength):]]

import plotly
import plotly.offline as py
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *

##
## Visualise
##
def visualise(data, periods, count):

    plotly.offline.init_notebook_mode() # run at the start of every ipython notebook

    csticks = data.values[0:count:,:periods*4].ravel().reshape(-1,4)

    fig = FF.create_candlestick(csticks[:,0], csticks[:,1], csticks[:,2], csticks[:,3])

    py.iplot(fig, filename='jupyter/simple-candlestick', validate=True)
    
    
##
## Save data
##
def save_hdf(data, dataset, hdfStore):
    hdfStore.put(dataset["name"], data, format='table')
    print "Saved data to HDFStore: /" + dataset["name"]
    return data

def save_csv(data, filename):
    data.to_csv( filename, mode="w", header=False, index=False)
    print "Saved data to " + filename
    
    return data