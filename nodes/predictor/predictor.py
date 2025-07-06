import json
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')

# Read configurations
config = json.loads(input())

def load():
    try:
        from sklw import SKLW
        return SKLW(path=config['path'])
    except:
        try:
            from dnnctf import DNNCTF
            return DNNCTF(path=config['path'], load=True)
        except:
            return None

model = load()

while True:
    try:
        # Read request
        features = pd.read_json(input(), orient='values')

        if model is None:
            model = load()
        if model is None:
            raise Exception('Cannot find model.')

        model.update()

        prediction = model.predict(features)

        # Safely convert to JSON serializable
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()

        print(json.dumps(prediction))

    except Exception as e:
        print(json.dumps({ "error": str(e) }))
