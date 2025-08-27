# Feature Scaling Used to range the higher value between 0-1
# 1 standard scaler mean -0  standard deviation-1 
# from sklearn.preprocessing import StandardScaler , MinMaxScaler 
# scaler =StandardScaler()
# x_scaled = scaler.fit_transform()
# scaler=MinMaxScaler()
# x_scaled =scaler.fit_transform()
import pandas as pd 
from sklearn.preprocessing import  StandardScaler ,MinMaxScaler 
from sklearn.model_selection import train_test_split
data= {
     'StudyHours':[1,2,3,4,5],
     'TestScore':[40,50,60,70,80]
     
 }
df=pd.DataFrame(data)
# standard scaler
standard_scaler =StandardScaler()
standard_scaled=standard_scaler.fit_transform(df)
print("\nstandard_scaler")
print(pd.DataFrame(standard_scaled,columns=['StudyHours','TestScore']))
minmax_scaler=MinMaxScaler()
minmax_scaled=minmax_scaler.fit_transform(df)
print("\nminmax_scaler")
print(pd.DataFrame(minmax_scaled,columns=['StudyHours','TestScore']))

# formula  z=x-mean/standard deviation this gives standard scalere to trange the values  and X-Xmn/Xmax-Xmin this give minmax
# double brackets means we wants to put data as dataframe and also it return series 2d Array
X=df[["StudyHours"]]
Y=df[["TestScore"]]
x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.2,random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)