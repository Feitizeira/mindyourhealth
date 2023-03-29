import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "data": [
      {
        "Column2": "example_value",
        "HighBP": 1,
        "HighChol":1,
        "CholCheck": 1,
        "BMI": 25,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 2,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 9,
        "Education": 6,
        "Income": 2
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}

def azresult(data):
    body = str.encode(json.dumps(data))

    url = 'http://adb39b8c-9501-4eb1-bf51-ba59a35f1ebb.westeurope.azurecontainer.io/score'


    headers = {'Content-Type':'application/json'}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        return result
    
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

