# R² = 1 (100%) → Perfect fit (model explains all variations in Y).

# R² = 0 → Model does no better than just predicting the mean of Y.

# R² < 0 → Model is worse than the mean (bad fit).

from sklearn.metrics import mean_absolute_error ,mean_squared_error,r2_score
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv("studyhours.csv")
X=data[['StudyHours']]
y=data['MarksObtained']
model=LinearRegression()
model.fit(X,y)
predicted_score=model.predict(X)
mae=mean_absolute_error(y,predicted_score)
mse=mean_squared_error(y,predicted_score)
rmse=np.sqrt(mse)
r2 =r2_score(y,predicted_score)
print("mae:", round(mae,2))
print("mse:" ,round(mse,2))
print("rmse:", round(rmse,2))
print("r2 score",round(r2,2)) # close to 1

#histogtram
plt.figure(figsize=(10,6))
plt.hist(data["MarksObtained"],bins=30 ,color='skyblue',edgecolor='black')
plt.title("distribution of final exam score")
plt.xlabel("final exam score")
plt.ylabel("Number of stidents")
plt.grid(True)
plt.show()

#scattered Plot 
plt.figure(figsize=(10,6))
plt.scatter(X,y ,color='blue',label="Actual Score")
plt.plot(X,predicted_score,color='red',label='predicted_score')
plt.title("Model predicted vs actual score")
plt.xlabel("Study hours per week")
plt.ylabel("Final Output")
plt.grid(True)
plt.show()

newhours=9
predicted_new_score=model.predict([[newhours]])
print(f"predicted final score for {newhours} Hours is {predicted_new_score} ")