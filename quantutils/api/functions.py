import json
import pandas
import requests
from quantutils.api.bluemix import Token


class Functions:

    def __init__(self, credentials_store):
        self.credentials = credentials_store.getSecrets('functions_cred')
        self.token = Token(credentials_store).getToken()

    def call_function(self, name, args, debug=False):
        headers = {'Content-Type': 'application/json', 'Authorization': "".join(["Bearer ", self.token['access_token']])}
        url = "".join([self.credentials["endpoint"], name, "?blocking=true"])
        resp = requests.post(url=url, data=json.dumps(args), headers=headers)

        if (debug):
            print(resp.text)

        return pandas.read_json(json.dumps(json.loads(resp.text)["response"]["result"]), orient='split')
