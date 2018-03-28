
###################
## Bluemix Setup ##
###################

from io import StringIO
import requests
import json
import logmet

class Logger:
    def __init__(self, appname, credentials_file):

        self.appname = appname
        logging_cred = json.load(open(credentials_file))
        self.log = logmet.Logmet(
            logmet_host=logging_cred['logmet_host'],
            logmet_port=logging_cred['logmet_port'],
            space_id=logging_cred['space_id'],
            token=logging_cred['token']
        )

    def info(self, msg):
        print "".join(["INFO: ", msg])
        self.log.emit_log({'app_name': self.appname,'type': 'info','message': msg})


    def error(self, msg):
        print "".join(["ERROR: ", msg])
        self.log.emit_log({'app_name': self.appname,'type': 'error','message': msg})


    def debug(self, msg):
        print "".join(["DEBUG: ", msg])
        self.log.emit_log({'app_name': self.appname,'type': 'debug','message': msg})

class Metrics:
    def __init__(self, credentials_file):
        self.credentials = json.load(open(credentials_file))

    def send(self, data):
        headers = {}
        headers['Content-Type'] = 'application/json'
        headers['x-auth-scope-id'] = ''.join(['s-', self.credentials['space_id']])
        headers['x-auth-user-token'] = ''.join(['apikey ', self.credentials['token']])
        resp = requests.post(url=self.credentials['host'], headers=headers, data=json.dumps(data) )
        return resp


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
    