import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np 
data =pd.read_csv("student.csv")
X=data[['Hours']] #double brackets beacuse of 2d inputs
y=data['Score'] # target columns
model =LinearRegression()
model.fit(X,y)
predicted_score =model.predict(X)
#Evaluate
mae=mean_absolute_error(y,predicted_score)
mse=mean_squared_error(y,predicted_score)
rmse=np.sqrt(mse)

#show results
print("mean absoulte error is :", mae)
print("mean squared error is :", mse)
print("Root mean  squared  error is :", rmse)

new_Hour =float(input('enter an hour:'))
new_predict=model.predict([[new_Hour]])
print(f"prediction for {new_Hour} is score ={new_predict}")
