import numpy as np
import pandas as pd
import quantutils.dataset.pipeline as ppl

# Accepts a ONE-HOT ENCODED list of predictions


def evaluate(predictions, data_y, threshold=0.5, display=True):

    a = np.argmax(predictions, axis=1)
    b = np.argmax(data_y, axis=1)

    a = a[(predictions >= threshold).any(axis=1)]
    b = b[(predictions >= threshold).any(axis=1)]

    num = np.float32(np.sum(a == b))
    den = np.float32(b.shape[0])

    if display:
        print("Won : " + str(num))
        print("Lost : " + str(den - num))
        print("Total : " + str(den))
        print("Diff : " + str(num - (den - num)))
        print("Edge : " + str(100 * (num - (den - num)) / den) + "%")
        print("IR : " + str(((num - (den - num)) / den) * np.sqrt(den)))

    # recall = np.float32(b.shape[0]) / data_y.shape[0] # Number of Days traded
    #F_score = (2.0 * precision * recall) / (precision + recall)

    return num / den  # precision

# aggregatePredictions:
##
# Options
# 1 np.nanmean(results["test_predictions"], axis=0)
# 2 np.nanmean(results["test_predictions"]) ## Remove conflicting predictions across markets
# 3 (np.sum(np.sign(r-.5), axis=0)/40)+.5 ## Vote rather than average (removes effects of outliers). Voting will ignore the strength of the individual prediction.
# 4 (np.sum(np.sign(r-.5))/80)+.5  ## Vote and Remove conflicting predictions across markets
# 5 as #1 but don't score any entries that are in conflict
# 6 as #3 but don't score any entries that are in conflict


def aggregatePredictions(predictions_list, method='vote_unanimous_all'):
    a = predictions_list[0]
    if (method == 'mean_all'):  # Remove conflicting predictions across markets
        for predictions in predictions_list[1:]:
            a = a.add(predictions)
        a = a / len(predictions_list)
        result = a.mean(axis=1).to_frame()
    if (method == 'vote_majority'):  # Vote rather than average (removes effects of outliers) across all data. Voting will ignore the strength of the individual prediction.
        a = np.sign(a - .5)
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(predictions - .5))
        result = ((a.sum(axis=1) / (predictions_list[0].shape[1] * len(predictions_list) * 2)) + .5).to_frame()
    if (method == 'vote_unanimous_pred'):  # Vote and Remove conflicting predictions across sub-predictions
        a = np.sign(np.sign(a - .5).sum(axis=1))
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(np.sign(predictions - .5).sum(axis=1)))
        result = (a / (len(predictions_list) * 2) + .5).to_frame()
    if (method == 'vote_unanimous_markets'):  # Vote and Remove conflicting predictions across markets
        a = np.sign(a - .5)
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(predictions - .5))
        a = ((a.sum(axis=1) / (predictions_list[0].shape[1] * len(predictions_list) * 2)) + .5)
        result = a[a != .5].to_frame()
    if (method == 'vote_unanimous_all'):  # Vote and Remove conflicting predictions across sub-predictions and markets
        a = np.sign(np.sign(a - .5).sum(axis=1))
        for predictions in predictions_list[1:]:
            a = a.add(np.sign(np.sign(predictions - .5).sum(axis=1)))
        a = (a / ((len(predictions_list) * 2))) + .5
        result = a[a != .5].to_frame()
    return result

# Accepts a ONE-HOT ENCODED list of float predictions, returns an array of tradeframework signals


def toMatchedTradeSignals(predictions, data_y, threshold=0):
    signals = np.ones(len(data_y))
    a = np.argmax(predictions, axis=1)
    b = np.argmax(data_y, axis=1)
    signals[a != b] = -1
    signals[(predictions <= threshold).all(axis=1)] = 0
    return signals

# Accepts a ONE-HOT ENCODED list of float predictions, returns an array of tradeframework signals


def toTradeSignals(predictions, threshold=0):
    signals = np.ones(len(predictions))
    a = np.argmax(predictions, axis=1)
    signals[(a == 1)] = -1  # Set any DOWN signals to -1 (UP signals will pick up the default of 1)
    # Implement a threshold AND ensure any 0-values predictions results in a 0 signal
    signals[(predictions <= threshold).all(axis=1)] = 0
    return signals

# Proxy for dataset onehot


def onehot(predictions):
    return ppl.onehot(predictions)
