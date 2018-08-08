import numpy as np
import pandas as pd
import pytz
import json

from quantutils.api.auth import CredentialsStore
from quantutils.api.bluemix import CloudObjectStore
from quantutils.api.marketinsights import MarketInsights, Dataset
from quantutils.model.ml import Model

COS_BUCKET = "marketinsights-weights"

cred = CredentialsStore('api/cred')
cos = CloudObjectStore(cred)
mi = MarketInsights(cred)

class MIModelClient():

    modelId = None
    modelInstance = None

    def score(self, training_id, dataset, aggMethod=None):

        training_run = mi.get_training_run(training_id)
        if (not training_run):
            return "No Training Id found"
        if (not cos.keyExists(COS_BUCKET, training_id)):
            return "No trained weights found for this training id"

        model_id = training_run["model_id"]
        _, dataset_desc = mi.get_dataset_by_id(training_run["datasets"][0]) # TODO this is too heavyweight just to get the desc
        weights = cos.get_csv(COS_BUCKET, training_id)
        model = self.getModelInstance(model_id, dataset_desc["features"], dataset_desc["labels"])        
        index = pd.DatetimeIndex(dataset["index"], tz=pytz.timezone(dataset["tz"]))
        predictions = self.getPredictions(model, index.astype(np.int64) // 10**9, np.array(dataset["data"]), aggMethod, weights) 
        return json.loads(Dataset.csvtojson(pd.DataFrame(predictions, index), None, None, createId=False))

    def getModelInstance(self, model_id, features, labels):
        if (model_id is not self.modelId):
            self.modelInstance = self.createModelInstance(model_id, features, labels)
            self.modelId = model_id
        return self.modelInstance

    def createModelInstance(self, model_id, features, labels):
        model_config = mi.get_model(model_id)
        # Create ML model
        return Model(features, labels, model_config)

    # Function to take dates, dataset info for those dates
    def getPredictions(self, model, timestamps, dataset, aggMethod, weights):

        if aggMethod is None:
            aggMethod = lambda x: np.nanmean(x,axis=0)

        # Load timestamps from weights db (or load all weights data)
        wPeriods = weights["timestamp"].values
        tsPerPeriod = np.sum(wPeriods==wPeriods[0])

        # x = for each dataset timestamp, match latest available weight timestamp
        latestPeriods = np.zeros(len(timestamps)) 
        uniqueWPeriods = np.unique(wPeriods)  # q
        mask = timestamps>=np.min(uniqueWPeriods)
        latestPeriods[mask] = [uniqueWPeriods[uniqueWPeriods<=s][-1] for s in timestamps[mask]]

        # for each non-duplicate timestamp in x, load weights into model for that timestamp
        results = np.empty((len(dataset), tsPerPeriod))
        for x in np.unique(latestPeriods):
            # run dataset entries matching that timestamp through model, save results against original timestamps
            mask = latestPeriods==x
            predictions = model.predict(weights[wPeriods==x].values[:,1:], dataset[mask])
            results[mask] = predictions

        # TODO insert default aggregator here
        
        return results    