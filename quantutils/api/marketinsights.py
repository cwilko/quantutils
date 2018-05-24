import pandas
import pytz
import json
import requests
import time
import hashlib
import dateutil.parser as parser

class MarketInsights:
    
    def __init__(self, credentials_file):
        credentials = json.load(open(credentials_file))
        self.credentials = credentials
        
    def put_dataset(self, data, dataset_desc, market, debug=False):
        dataset = Dataset.csvtojson(data, dataset_desc, market)
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }        
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/datasets"])
        resp = requests.put(url=url, headers=headers, data=dataset)  
        if debug:
            print(resp.text)
        return json.loads(resp.text)

    def get_dataset(self, dataset_desc, market, debug=False):
        return self.get_dataset_by_id(Dataset.generateId(dataset_desc, market), debug)
    
    def get_dataset_by_id(self, dataset_id, debug=False):        
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  } 

        # TODO eliminate id, no need.       
        query = { \
                 'where': { \
                          'id': dataset_id, \
                          } \
                }
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/datasets?filter=",json.dumps(query)])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print(resp.text)
        dataset = json.loads(resp.text)[0]
        return [Dataset.jsontocsv(dataset), dataset["dataset_desc"]]

    def put_predictions(self, data, market, modelId, throttle=10, sleep=2, debug=False, update=False):

        ## Throttle API calls (throttle = Number of calls/sec)
        if (throttle is not None):
          i = 0
          for j in range(throttle, len(data)+throttle, throttle):
            print("".join(["Sending chunk ", str(j//throttle)," of ", str((len(data)//throttle)+1)]))
            res = self.put_predictions(data[i:j], market, modelId, throttle=None, debug=debug, update=update)
            if ("error" in res):
              return res
            time.sleep(sleep)
            i = j
          return {"success": True}
        else:

          ## POST Prediction object to API
          data = Predictions.csvtojson(data, market, modelId)
          headers = { \
                     'X-IBM-Client-Id': self.credentials["clientId"], \
                     'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                     'content-type': 'application/json' \
                    }        
                     
          if (update):            
            url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions"])
            for prediction in data:               
              resp = requests.put(url=url, headers=headers, data=json.dumps(prediction))
              if debug:
                print(resp.text)
          else:
            url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions"])
            resp = requests.post(url=url, headers=headers, data=json.dumps(data))

          if debug:
              print(resp.text)
          return json.loads(resp.text)

    # TODO : Deprecated
    def get_predictions(self, market, modelId, start=None, end=None, debug=False):
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }

        query = Predictions.getQuery(market, modelId, start, end)

        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions?filter=",json.dumps(query)])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print(resp.text)
        return Predictions.jsontocsv(json.loads(resp.text))

    def delete_predictions(self, market, modelId, start=None, end=None, debug=False):
          
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }
        
        data = Predictions.csvtojson(self.get_predictions(market, modelId, start, end, debug), market, modelId)
        if (len(data)>0):
          for prediction in data:
            url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/predictions/", prediction["id"]])
            resp = requests.delete(url=url, headers=headers, data=json.dumps(prediction))
            if debug:
                print(prediction)
                print(resp.text)

          return resp
        else:
          return []

    def put_model(self, data,debug=False):
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }        
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/models"])
        resp = requests.put(url=url, headers=headers, data=json.dumps(data))  
        if debug:
            print(resp.text)
        return json.loads(resp.text)
    
    def get_model(self, modelId, debug=False):        
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }         
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/models/", modelId])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print(resp.text)
        return json.loads(resp.text)

    def put_training_run(self, data, debug=False):
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'content-type': 'application/json' \
                  }        
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/training_runs"])
        resp = requests.put(url=url, headers=headers, data=json.dumps(data))  
        if debug:
            print(resp.text)
        return json.loads(resp.text)

    def get_training_run(self, training_run_id, debug=False):        
        headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }         
        url = "".join([self.credentials["endpoint"],"/miol-prod/api/v1/training_runs/", training_run_id])
        resp = requests.get(url=url, headers=headers) 
        if debug: 
            print(resp.text)
        return json.loads(resp.text)

    def get_score(self, featureSet, model_id, training_id):
      headers = { \
                   'X-IBM-Client-Id': self.credentials["clientId"], \
                   'X-IBM-Client-Secret': self.credentials["clientSecret"], \
                   'accept': 'application/json' \
                  }
      

      pass

class Dataset:

    @staticmethod
    def csvtojson(csv, dataset_desc, market):
        obj = {"id":Dataset.generateId(dataset_desc, market), "dataset_desc":dataset_desc, "market":market}
        obj["data"] = csv.values.tolist()
        obj["tz"] = csv.index.tz.zone
        obj["index"] = [date.isoformat() for date in csv.index.tz_localize(None)] # Remove locale 
        return json.dumps(obj)

    @staticmethod
    def jsontocsv(jsonObj):
        return pandas.DataFrame(jsonObj["data"], index=pandas.DatetimeIndex(jsonObj["index"], name="Date_Time", tz=pytz.timezone(jsonObj["tz"])))   

    @staticmethod
    def generateId(dataset_desc, market):
      return hashlib.md5("".join([market, str(dataset_desc["pipeline"]), str(dataset_desc["features"]), str(dataset_desc["labels"])]).encode('utf-8')).hexdigest()

class Predictions:

    @staticmethod
    def csvtojson(data, market, modelId):
        data.index = data.index.tz_localize(None)
        obj = [{ \
          "id": hashlib.md5("".join([modelId,"_",market,"_",i.isoformat()]).encode('utf-8')).hexdigest(), \
          "market":market, \
          "model_id":modelId, \
          "timestamp":i.isoformat(), \
          "data":data.loc[i].values.tolist()} for i in data.index]
        return obj

    @staticmethod
    def jsontocsv(jsonObj):
        idx = pandas.DatetimeIndex([prediction["timestamp"] for prediction in jsonObj])
        return pandas.DataFrame([prediction["data"] for prediction in jsonObj], idx).sort_index()

    @staticmethod
    def getQuery(market, modelId, start, end):
        query = { \
          'where': { \
            'market': market, \
            'model_id': modelId \
          } \
        }

        if (start is not None and end is not None):
          query["where"]["timestamp"] = { 'between': [parser.parse(start).isoformat(), parser.parse(end).isoformat()] }
        else:            
          if (start is not None):
            query["where"]["timestamp"] = { 'gte':parser.parse(start).isoformat() }

          if (end is not None):
            query["where"]["timestamp"] = { 'lte':parser.parse(end).isoformat() }

        return query