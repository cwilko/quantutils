import pandas
import numpy
import pytz

##
# Merge data
##


def merge(newData, existingData):
    # print("Merging data...")
    return existingData.combine_first(newData).astype(existingData.dtypes.to_dict())

##
# Shuffle data
##


def shuffle(data):
    return data.sample(frac=1).reset_index(drop=True)

##
# Concatenate Columns
##


def concat(data1, data2, axis=1):
    print("Concatenating features %s with classifications %s" % (data1.shape, data2.shape))
    return pandas.DataFrame(numpy.concatenate([data1.values, data2.values], axis=axis), data1.index)

##
# Crop
##


def cropDate(data, start="1979-01-01", end="2050-01-01"):
    return data[start:end]


def cropTime(data, start, end):
    return data.between_time(start, end, include_start=True, include_end=False)

##
# Resample
##


def resample(data, sample_unit, debug=False):
    if debug:
        print("Resampling to %s periods" % sample_unit)

    func = {'Open': 'first', 'High': lambda x: x.max(skipna=True), 'Low': lambda x: x.min(skipna=True), 'Close': 'last'}
    for col in data.columns.difference(["Open", "High", "Low", "Close"]):
        func[col] = 'sum'
    return data.resample(sample_unit).agg(func)[data.columns]

##
# Remove Missing Data (NaN)
##


def removeNaNs(data):
    return data.dropna()
##
# Convert to Feature Matrix
##


def toFeatureSet(data, feature_periods):
    n = data.values.shape[1] * feature_periods
    return pandas.DataFrame(reshape(data, n), data.index[range(0, len(data), feature_periods)])


def reshape(ts, n):
    return numpy.reshape(ts.values[0:ts.values.size // n * n // ts.values.shape[1]], (ts.values.size // n, n))

##
# Encode classification
##


def encode(data, encoding):
    nanIndex = data.isnull().any(axis=1)
    if (encoding == "binary"):
        df = pandas.DataFrame((data.values[:, -1] > data.values[:, 0]).astype(float), data.index)
    if (encoding == "one-hot"):
        df = pandas.DataFrame(numpy.column_stack
                              ([(data.values[:, -1] > data.values[:, 0]).astype(float),
                                  (data.values[:, 0] > data.values[:, -1]).astype(float)]), data.index)

    df.values[nanIndex.values] = numpy.nan

    return df

##
# Onehot
##


def onehot(labels, threshold=0):
    if (labels.shape[1] > 1):
        return labels
    a = numpy.zeros(len(labels))
    b = numpy.zeros(len(labels))
    a[(labels >= threshold).flatten()] = labels[labels >= threshold]
    b[((1 - labels) > threshold).flatten()] = 1 - labels[(1 - labels) > threshold]
    return numpy.array([a, b]).T


def localize(data, sourceTZ, targetTZ):
    #print("Converting from " + sourceTZ + " to " + targetTZ)
    timezone = pytz.timezone(targetTZ)
    if not data.tz:
        data = data.tz_localize(sourceTZ, level="Date_Time", ambiguous='NaT')
    data = data.tz_convert(timezone, level="Date_Time")
    data = data[data.index.get_level_values("Date_Time").notnull()]  # Remove any amiguous timezone rows
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
    if (allowZero != True):
        X = (X * 2) - 1
    # Add additional feature to reflect the percentage range over the Open (scaled up to be between 0 and 1)
    return pandas.DataFrame(numpy.hstack((X, numpy.column_stack(scale.T * 100 / data[0].values))), data.index)

##
## Split (Train/Val/Test)
##


def splitRows(data, train=.6, val=.2, test=.2):
    idx = numpy.arange(0, len(data)) / float(len(data))
    msk1 = data[idx < train]
    msk2 = data[(idx >= train) & (idx < (train + val))]
    msk3 = data[(idx >= (train + val))]
    return [msk1, msk2, msk3]

# Split with absolute value (of test set)


def splitRowsAbs(data, testSetLength):
    return [data[:-(testSetLength)], data[-(testSetLength):]]

# Split columns after column number (used for separating features/classes)


def splitCol(data, col_num):
    return data.values[:, :col_num], data.values[:, col_num:]

##
# Interleave two datasets
##


def interleave(data1, data2):
    import toolz
    return pandas.DataFrame(toolz.interleave([data1.values, data2.values]))

##
# (De)Interleave two datasets
##


def deinterleave(data):
    data1 = data[::2]
    data2 = data[1::2]
    return [pandas.DataFrame(data1), pandas.DataFrame(data2)]

##
# Get a Dataframe that has the index intersect of two input Dataframes
##


def intersect(data1, data2):
    isect = data1.index.intersection(data2.index)
    return [data1.loc[isect], data2.loc[isect], isect]

##
# Save data
##


def save_hdf(data, bucket, hdfStore):
    hdfStore.put(bucket, data, format='table')
    print("Saved data to HDFStore: /" + bucket)
    return data


def save_csv(data, filename):
    data.to_csv(filename, mode="w", header=False, index=False)
    print("Saved data to " + filename)

    return data
