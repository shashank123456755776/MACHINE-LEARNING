# encoding category data menas data in words chnage into numerical such that models can able to understands 
# its two type label encoding and one hot encoding 

# Lable encoding 
# male -->0 Female--->1  means onlt two data (category)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Read the CSV file
df = pd.read_csv("sample.csv")

# Create a copy of the DataFrame
df_label = df.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to the 'Gender' column
df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"])

# Print the original and encoded Gender columns
# double brackets means to select multiple columns as dataframe 
print(df_label[["Gender", "Gender_Encoded"]])
