# step 1 Load and understanding Data
# import pandas as pd 
# df =pd.read_csv("studentsuccess.csv")
# print("print first five data rows")
# print(df.head())

# print("dataset shape")
# print(f'Rows:{df.shape[0]},Columns:{df.shape[1]}')
# print("dataset info")
# print(df.info())
# print("summaries statics:")
# print(df.describe(include="all"))
# print("missing values:")
# print(df.isnull().sum())
# df.info() → कितनी values missing हैं, दिखाता है।

# df.describe() → count कम दिखेगा अगर NaN हैं।

# df.isnull().sum() → exact कितने NaN हैं बताता है।

# जब हम df.describe(include="all") चलाते हैं:

# इसके output में categorical columns के लिए यह stats आता है:

# count → कितनी total entries हैं

# unique → कितने different (distinct) values हैं

# top → कौन सा value सबसे ज्यादा बार आया

# freq → top वाला कितनी बार आया

# step 2  Data Preprocessing (Clean Data)
# categorial to numerical values then use label encoder or get dummies
# and if any missing values hai to isnull() ka istammal karenge
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
df=pd.read_csv("studentsuccess.csv")
print("missing values in each columns:")
print(df.isnull().sum())
le=LabelEncoder()
df["Internet"] =le.fit_transform(df["Internet"])
df["Passed"]=le.fit_transform(df["Passed"])

print("After encoding")
print(df.head())
print("datatype after cleaning")
print(df.dtypes)

# ager bas yes/no ya True/False hai coulmns to hamm label encoder ka use  karunga jo boolean data ko 0 and 1 mai chnage kar dega 
# ager more cities ho jsime bhoit option ho to hmm get dummies ka use karta hai jo use treu nad false maim  change karta hai then label encoder ka use kar ke numercial value mai chnage kar lete hai
# ager koi null values hoga then use drop ya fill kar denge

# step 3 Feature scaling 
# data ko same scale and same range pe lane ke liye hmm standard scaler or minmax scaler
# inka estamml karne ka purpose ye hai ki hamara coulms ka data ek range mai aa jaye 
# step4 spilt the data--apne model ko khuch data pe train karnge and then test karenge ki hamara model khuch sikha ki ni
# step5 model ko actual mai train karenge ...logistic,linear use kar sakte hai
# step6 prediction dekhenge
# step7 model evaluations
# step8 plot graph and visuliaze results
# step9 improve experiments
# step10 apna code dekhna also write .readme file

