import argparse
import numpy as np
import json
from pathlib import Path
import joblib
import pandas as pd

parent_dir = Path('__file__').parent

model = parent_dir / 'artifactory' / "land_prediction.pkl"

def load_model():
    if not model.exists():
       raise FileNotFoundError("Model not found please run train.py")
    return joblib.load(model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str,  help="please parse input values")
    parser.add_argument("--csv", type=str, help="provide csv file")
    args = parser.parse_args()
    if args.input:
       try:
           features = json.loads(args.input)
       except json.JSONDecodeError:
           raise ValueError("Invalid Input")
       data= np.array(features).reshape(1,-1)
       model = load_model()
       pred = model.predict(data)
       print(json.dumps({"predction" : pred.tolist()}))
    elif args.csv:
         data = pd.read_csv(args.csv)
         model = load_model()
         pred = model.predict(data)
         print(json.dumps({"predcitions" : pred.tolist()}))
    else:
       print("please provide input or csv file")
if __name__ == "__main__":
    main()