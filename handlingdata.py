import pandas as pd 
data ={
    "Name":["shashank","rohit","sohan","kunal","mehak"],
    "Age":[23,None,24,25,None],
    "Salary":[40000,None,50000,700000,None]
}
df=pd.DataFrame(data)
print(df)
# It tells about the sum of row of how many data missing in each column
print(df.isnull().sum())
# precentage to find percentage data missings 
print(df.isnull().mean()*100)
df_dropna=df.dropna()
print(df_dropna)
# how to fill the missing data we try to fill  as mean 
# if we working with numerical data then works with mean and median
# if we working with categories data then works with Mode
df["Age"].fillna(df["Age"].mean(),inplace=True)
df["Salary"].fillna(df["Salary"].mean(),inplace=True)
print(df)
