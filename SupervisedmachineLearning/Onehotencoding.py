import pandas as pd 
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("sample.csv")
df_label=df.copy()
le=LabelEncoder()
df_encoded = pd.get_dummies(df_label ,columns=["city"])
print(df_encoded)
df_encoded["city_delhi"] =le.fit_transform(df_encoded["city_delhi"])
df_encoded["city_mumbai"] =le.fit_transform(df_encoded["city_mumbai"])
df_encoded["city_rajasthan"] =le.fit_transform(df_encoded["city_rajasthan"])
print(df_encoded[["city_delhi","city_mumbai","city_rajasthan"]])
