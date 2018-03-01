import pandas
import pytz
import json
import requests
import time

class MarketInsights:
    
    def __init__(self, credentials_file):
        credentials = json.load(open(credentials_file))
        self.credentials = credentials
        
    def put_dataset(self, data, market, pipeline_id, debug=False):
        dataset = Dataset.csvtojson(data, market, pipeline_id)
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }        
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/datasets"])
        resp = requests.put(url=url, headers=headers, data=dataset)  
        if debug:
            print resp.text
        return json.loads(resp.text)
    
    def get_dataset(self, market, pipelineId, debug=False):        
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }        
        query = { \
                 'where': { \
                          'id': "".join([pipelineId,"_",market]), \
                          } \
                }
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/datasets?filter=",json.dumps(query)])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print resp.text
        return Dataset.jsontocsv(json.loads(resp.text)[0])

    def put_predictions(self, data, market, modelId, throttle=None, sleep=2, debug=False):

        ## Throttle API calls (throttle = Number of calls/sec)
        if (throttle is not None):
          i = 0
          for j in range(throttle, len(data)+throttle, throttle):
            res = self.put_predictions(data[i:j], market, modelId, throttle=None, debug=debug)
            if ("error" in res):
              return res
            time.sleep(sleep)
            i = j

        ## POST Prediction object to API
        data = Predictions.csvtojson(data, market, modelId)
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }        
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions"])
        resp = requests.post(url=url, headers=headers, data=data)  

        if debug:
            print resp.text
        return json.loads(resp.text)

    def get_predictions(self, market, modelId, start, end, debug=False):
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }        
        query = { \
                 'where': { \
                          'id': "".join([modelId,"_",market]), \
                          } \
                }
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions?filter=",json.dumps(query)])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print resp.text
        return Predictions.jsontocsv(json.loads(resp.text))

class Dataset:

    @staticmethod
    def csvtojson(csv, market, pipeline_id):
        obj = {"id":"".join([pipeline_id,"_",market]), "market":market, "pipelineID":pipeline_id}
        obj["data"] = csv.values.tolist()
        obj["tz"] = csv.index.tz.zone
        obj["index"] = [date.isoformat() for date in csv.index.tz_localize(None)] # Remove locale 
        return json.dumps(obj)

    @staticmethod
    def jsontocsv(jsonObj):
        return pandas.DataFrame(jsonObj["data"], index=pandas.DatetimeIndex(jsonObj["index"], name="Date_Time", tz=pytz.timezone(jsonObj["tz"])))   

class Predictions:

    @staticmethod
    def csvtojson(data, market, modelId):
        data.index = data.index.tz_localize(None)
        obj = [{ \
          "market":market, \
          "model_id":modelId, \
          "timestamp":i.isoformat(), \
          "data":data.loc[i].values.tolist()} for i in data.index]
        return json.dumps(obj)

    @staticmethod
    def jsontocsv(jsonObj):
        return pandas.DataFrame(jsonObj["data"], index=pandas.DatetimeIndex(jsonObj["index"], name="Date_Time", tz=pytz.timezone(jsonObj["tz"]))) 