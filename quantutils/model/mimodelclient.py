import numpy as np
import pandas as pd
import pytz
import json

from quantutils.api.bluemix import CloudObjectStore
from quantutils.api.marketinsights import MarketInsights, Dataset
from quantutils.model.ml import Model

COS_BUCKET = "marketinsights-weights"


class MIModelClient():

    modelId = None
    modelInstance = None

    def __init__(self, cred):
        self.cos = CloudObjectStore(cred)
        self.mi = MarketInsights(cred)

    def score(self, training_id, dataset):

        training_run = self.mi.get_training_run(training_id)
        if (not training_run):
            return "No Training Id found"
        if (not self.cos.keyExists(COS_BUCKET, training_id)):
            return "No trained weights found for this training id"

        model_id = training_run["model_id"]
        _, dataset_desc = self.mi.get_dataset_by_id(training_run["datasets"][0])  # TODO this is too heavyweight just to get the desc
        weights = self.cos.get_csv(COS_BUCKET, training_id)
        model = self.getModelInstance(model_id, dataset_desc["features"], dataset_desc["labels"])
        index = pd.DatetimeIndex(dataset["index"], tz=pytz.timezone(dataset["tz"]))
        predictions = self.getPredictions(model, index.astype(np.int64) // 10**9, np.array(dataset["data"]), weights)
        return json.loads(Dataset.csvtojson(pd.DataFrame(predictions, index), None, None, createId=False))

    def getModelInstance(self, model_id, features, labels):
        if (model_id is not self.modelId):
            self.modelInstance = self.createModelInstance(model_id, features, labels)
            self.modelId = model_id
        return self.modelInstance

    def createModelInstance(self, model_id, features, labels):
        model_config = self.mi.get_model(model_id)
        # Create ML model
        return Model(features, labels, model_config)

    # Function to take dates, dataset info for those dates
    def getPredictions(self, model, timestamps, dataset, weights):

        # Load timestamps from weights db (or load all weights data)
        wPeriods = weights["timestamp"].values
        tsPerPeriod = np.sum(wPeriods == wPeriods[0])

        # x = for each dataset timestamp, match latest available weight timestamp
        latestPeriods = np.zeros(len(timestamps))
        uniqueWPeriods = np.unique(wPeriods)  # q
        mask = timestamps >= np.min(uniqueWPeriods)
        latestPeriods[mask] = [uniqueWPeriods[uniqueWPeriods <= s][-1] for s in timestamps[mask]]

        # for each non-duplicate timestamp in x, load weights into model for that timestamp
        results = np.empty((len(dataset), tsPerPeriod))
        for x in np.unique(latestPeriods):
            mask = latestPeriods == x
            if (x == 0):
                predictions = np.zeros((len(dataset[mask]), tsPerPeriod))
            else:
                # run dataset entries matching that timestamp through model, save results against original timestamps
                predictions = model.predict(weights[wPeriods == x].values[:, 1:], dataset[mask])
                predictions = predictions.reshape(tsPerPeriod, len(dataset[mask])).T  # WARNING : ONLY WORKS FOR SINGLE LABEL DATA
            results[mask] = predictions

        #results = np.nanmean(results, axis=0)

        return results

    # This version of the function returns predictions from each trained model that is associated with all previous time periods - rather than a single
    # prediction for the current (or most recent) time period.
    # The results from using this tend to be far more stable, but ultimately lower than predictions from models associated with recent time periods.
    def _getPredictions_Historical(self, model, timestamps, dataset, weights):

        # Load timestamps from weights db (or load all weights data)
        wPeriods = weights["timestamp"].values
        tsPerPeriod = np.sum(wPeriods == wPeriods[0])

        # x = for each dataset timestamp, match latest available weight timestamp
        latestPeriods = np.zeros(len(timestamps))
        uniqueWPeriods = np.unique(wPeriods)  # q
        mask = timestamps >= np.min(uniqueWPeriods)
        latestPeriods[mask] = [uniqueWPeriods[uniqueWPeriods <= s][-1] for s in timestamps[mask]]

        # for each non-duplicate timestamp in x, load weights into model for that timestamp
        results = [np.array([])] * len(dataset)
        for x in np.unique(latestPeriods):
            # print(x)

            mask = [i for i in range(len(results)) if latestPeriods[i] >= x]
            predictions = model.predict(weights[wPeriods == x].values[:, 1:], dataset[mask])
            scores = predictions.reshape(tsPerPeriod, len(dataset[mask])).T  # WARNING : ONLY WORKS FOR SINGLE LABEL DATA

            for i in range(0, len(mask)):
                results[mask[i]] = np.append(results[mask[i]], scores[i])

            #results[mask] = mlutils.aggregatePredictions([pd.DataFrame(scores)], "vote_unanimous_pred").values

        #results = np.nanmean(results, axis=0)

        return results
