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
    print(
        "Concatenating features %s with classifications %s" % (data1.shape, data2.shape)
    )
    return pandas.DataFrame(
        numpy.concatenate([data1.values, data2.values], axis=axis), data1.index
    )


##
# Crop
##


def cropDate(data, start="1979-01-01", end="2050-01-01"):
    return data[start:end]


def cropTime(data, start=None, end=None):
    if not start:
        start = "00:00"
    if not end:
        end = "23:59:59"
    return data.between_time(start, end, inclusive="left")


##
# Resample
##


def resample(data, sample_unit, debug=False):
    if debug:
        print("Resampling to %s periods" % sample_unit)

    func = {
        "Open": "first",
        "High": lambda x: x.max(skipna=True),
        "Low": lambda x: x.min(skipna=True),
        "Close": "last",
    }
    for col in data.columns.difference(["Open", "High", "Low", "Close"]):
        func[col] = "sum"
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
    return pandas.DataFrame(
        reshape(data, n), data.index[range(0, len(data), feature_periods)]
    )


def reshape(ts, n):
    return numpy.reshape(
        ts.values[0 : ts.values.size // n * n // ts.values.shape[1]],
        (ts.values.size // n, n),
    )


##
# Encode classification
##


def encode(data, encoding, negative=0):
    nanIndex = data.isnull().any(axis=1)
    if encoding == "binary":
        # Is close (index -1) higher that open (index 0)
        df = pandas.DataFrame(
            (data.values[:, -1] > data.values[:, 0]).astype(float), data.index
        )
    if encoding == "one-hot":
        df = pandas.DataFrame(
            numpy.column_stack(
                [
                    (data.values[:, -1] > data.values[:, 0]).astype(float),
                    (data.values[:, 0] > data.values[:, -1]).astype(float),
                ]
            ),
            data.index,
        )

    df[df == 0] = negative
    df.values[nanIndex.values] = numpy.nan

    return df


def encode_single(data, encoding, negative=0):
    nanIndex = data.isnull().any(axis=1)
    if encoding == "binary":
        # Is close (index -1) higher that open (index 0)
        df = pandas.DataFrame((data["Close"] > data["Open"]).astype(float), data.index)
    elif encoding == "one-hot":
        df = pandas.DataFrame(
            numpy.column_stack(
                [
                    (data["Close"] > data["Open"]).astype(float),
                    (data["Open"] > data["Close"]).astype(float),
                ]
            ),
            data.index,
        )

    df[df == 0] = negative
    df.values[nanIndex.values] = numpy.nan

    return df


##
# Onehot
# Convert a list where values are probability of a 1, to an n x 2 matrix where first column is p(0) and second is p(1)
##


def onehot(labels, threshold=0):
    if labels.shape[1] > 1:
        return labels
    a = numpy.zeros(len(labels))
    b = numpy.zeros(len(labels))
    a[(labels >= threshold).flatten()] = labels[labels >= threshold]
    b[((1 - labels) > threshold).flatten()] = 1 - labels[(1 - labels) > threshold]
    return numpy.array([b, a]).T


def localize(data, sourceTZ, targetTZ):
    # print("Converting from " + sourceTZ + " to " + targetTZ)
    timezone = pytz.timezone(targetTZ)
    if not hasattr(data.index, "tz"):
        data = data.tz_localize(sourceTZ, level="Date_Time", ambiguous="NaT")
    elif not sourceTZ == "UTC":
        raise Exception("Unknown sourceTZ")

    data = data.tz_convert(timezone, level="Date_Time")
    data = data[
        data.index.get_level_values("Date_Time").notnull()
    ]  # Remove any amiguous timezone rows
    return data


#
# Normalise (Candlesticks)
#


def normaliseCandlesticks(data, allowZero=True):
    X = data.values
    Xmax = X.max(axis=1)[numpy.newaxis].T
    Xmin = X.min(axis=1)[numpy.newaxis].T
    range = Xmax - Xmin
    X = (X - Xmin) / range
    if not allowZero:
        X = (X * 2) - 1

    # Feature to reflect the range as a percent of the Open price for that period.

    # Calibrated to provide a normalised range percent between 0 and 1 (for logged prices)
    RANGE_CONSTANT = 300
    range_pct = range.T * RANGE_CONSTANT / data[0].values
    return pandas.DataFrame(
        numpy.hstack((X, numpy.column_stack(range_pct))), data.index
    )


def normaliseCandlesticks_single(data, allowZero=True):
    X = data.values
    Xmax = X.max(axis=1)[numpy.newaxis].T
    Xmin = X.min(axis=1)[numpy.newaxis].T
    range = Xmax - Xmin
    X = (X - Xmin) / range
    if not allowZero:
        X = (X * 2) - 1

    # Feature to reflect the range as a percent of the Open price for that period.

    # Calibrated to provide a normalised range percent between 0 and 1 (for logged prices)
    # TODO: These are log prices so this should just be the log diff between close and open?
    RANGE_CONSTANT = 300
    range_pct = range.T * RANGE_CONSTANT / data.iloc[:, 0].values
    return pandas.DataFrame(
        numpy.hstack((X, numpy.column_stack(range_pct))),
        data.index,
        columns=["Open", "High", "Low", "Close", "Range"],
    )


##
## Split (Train/Val/Test)
##


def splitRows(data, train=0.6, val=0.2, test=0.2):
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
    hdfStore.put(bucket, data, format="table")
    print("Saved data to HDFStore: /" + bucket)
    return data


def save_csv(data, filename):
    data.to_csv(filename, mode="w", header=False, index=False)
    print("Saved data to " + filename)

    return data


def compare(df1: object, df2: object) -> object:
    "Compare two dataframes"

    try:
        return df1.compare(df2)
    except:
        print(
            "Could not compare dataframes with pandas. Retrying with other methods..."
        )

    if not df1.equals(df2):
        print("Pandas shows dataframes are not equal")

    print("Length of df1: " + str(len(df1)))
    print("Length of df2: " + str(len(df2)))
    if len(df1) != len(df2):
        print("Lengths do not match!")

        if len(df1) > len(df2):
            x = df1.join(df2, lsuffix="_df1", rsuffix="_df2")
            x = x[x.isna().any(axis=1)]
        else:
            x = df2.join(df1, lsuffix="_df2", rsuffix="_df1")
            x = x[x.isna().any(axis=1)]
        print("Showing differing rows: ")
        display(x)

    else:
        print("Lengths match")

        print("Showing element-wise comparison")
        x = df1[df1 == df2]
        display(x)
        return x
