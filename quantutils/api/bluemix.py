
###################
## Bluemix Setup ##
###################

from io import StringIO
import requests
import json
import logmet

def get_metrics_client(metrics_cred):
    #metrics = logmet.Logmet(
    #    logmet_host='metrics.ng.bluemix.net',
    #    logmet_port=9095,
    #    space_id='xxx',
    #    token='xxx'
    #)
    return

def get_logging_client(logging_cred):
    return logmet.Logmet(
        logmet_host=logging_cred['logmet_host'],
        logmet_port=logging_cred['logmet_port'],
        space_id=logging_cred['space_id'],
        token=logging_cred['token']
    )

class ObjectStore:
    
    def __init__(self, credentials_file):
        credentials = json.load(open(credentials_file))
        self.load_obj_storage_token(credentials)
    
    def load_obj_storage_token(self, obj_storage_cred):
    
        url = ''.join([obj_storage_cred['auth_url'], '/v3/auth/tokens'])
        data = {'auth': {'identity': {'methods': ['password'],
                'password': {'user': {'name': obj_storage_cred['username'],'domain': {'id': obj_storage_cred['domainId']},
                'password': obj_storage_cred['password']}}}}}
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(url=url, data=json.dumps(data), headers=headers)
        resp_body = resp.json()
        for e1 in resp_body['token']['catalog']:
            if(e1['type']=='object-store'):
                for e2 in e1['endpoints']:
                            if(e2['interface']=='public'and e2['region']=='dallas'):
                                endpoint_url = e2['url']
        token = resp.headers['x-subject-token']

        self.endpoint_url = endpoint_url
        self.token = token
        
    def put_file(self, container, local_file_name, filename):  
        """This functions returns a StringIO object containing
        the file content from Bluemix Object Storage V3."""

        f = open(local_file_name,'r')    
        headers = {'X-Auth-Token': self.token, 'accept': 'application/json'}
        url = "".join([self.endpoint_url, "/", container, "/", filename])
        resp = requests.put(url=url, headers=headers, data = f.read() )
        print resp.text

    def get_file(self, container, filename):
        """This functions returns a StringIO object containing
        the file content from Bluemix Object Storage."""

        url = "".join([self.endpoint_url, "/", container, "/", filename])
        headers = {'X-Auth-Token': self.token, 'accept': 'application/json'}
        resp = requests.get(url=url, headers=headers)
        return StringIO(resp.text)
    