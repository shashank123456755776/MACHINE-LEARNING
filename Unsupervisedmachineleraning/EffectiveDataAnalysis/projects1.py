import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler ,LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report ,confusion_matrix 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv("studentsuccess.csv")
le=LabelEncoder()
df["Internet"]=le.fit_transform(df["Internet"])
df["Passed"]=le.fit_transform(df["Passed"])
features=["StudyHours","Attendance","PastScore","Sleephours"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])
X=df_scaled[features] #features
y=df_scaled["Passed"] #target 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) 
model =LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
# classification report
print("Classification Report:")
print(classification_report(y_test,y_predict))
# confusing matrix

confusion_matrix= confusion_matrix(y_test,y_predict)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("----predictResults......")
try:
    study_hours=float(input("enter study hours:"))
    attendance=float(input("enter attendance:"))
    past_score=float(input("enter past_score:"))
    sleep_hours=float(input("enter sleep_hours:"))
    user_input_df=pd.DataFrame([{
        "StudyHours":study_hours,
        "Attendance":attendance,
        "PastScore":past_score,
        "Sleephours":sleep_hours
    }])
    user_input_scaled=scaler.transform(user_input_df)
    prediction=model.predict(user_input_scaled)[0]
    result="pass" if prediction==1 else "Fail"
    print(f"prediction Based on input:{result}")
except Exception as e:
    print("an error occured" ,e)    