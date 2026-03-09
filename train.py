import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pathlib import Path
import joblib

def main ():
   #extract the data to readable format
   BASE_DIR = Path(__file__).parent
   data_path = BASE_DIR / "data" / "land.csv"
   try:
      data = pd.read_csv(data_path)
      print("success")
   except:
      print("error")
   #feature selction 
   x = data.drop(columns=['Price'])
   y = data['Price']

   #split the data
   x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


   #select the model/Algorithm
   model = LinearRegression()

   #Train the model 
   model.fit(x_train,y_train)

   directory = BASE_DIR / "artifacts"
   directory.mkdir(exist_ok=True)
   model_path = directory/"land_prediction.pkl"
   joblib.dump(model,model_path)

   #Evaluate the model
   predict = model.predict(x_test)
   mae =metrics.mean_absolute_error(y_test,predict)
   acc = model.score(x_test,y_test)
   accuracy = float(acc)
   print(mae,accuracy)

if __name__ == "__main__":
   main()




