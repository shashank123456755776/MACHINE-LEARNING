# MAE(MEAN ABSOLUTE ERROR) ,MSE ,RMSE
# MAE 
# EXAMPLE
# A -REALMARKS(90)-MODELPREDICT(85)-MISTAKE(5)
# B -REALMARKS(60)-MODELPREDICT(70)-MISTAKE(10)
# C -REALMARKS(80)-MODELPREDICT(70)-MISTAKE(10)
# D -REALMARKS(100)-MODELPREDICT(95)-MISTAKE(5)

# STEP 1-TAKE MISTAKE DIFFERENCES
# 2-REMOVE THE  - SIGN MEANS TAKE ABSOULTE
# 3-add mnistakes
# 4 divide add by total students
# means 30/4=7.5 average
# iska use karte hai tab jab hmm mistake pata lagna ho
# .........................................

# MSE(MEAN SQUARED ERROR)
# 5*5=25
#10*10=100
# 10*10=100
# 5*5=25
# ADD ALL =250/4=62.5(MEAN SQUARED ERRORS)

# RMSE(ROOT  MEAN SQUARED ERROR)
# SQAURE ROOT of 62.5

# practical code 
from sklearn.metrics import mean_absolute_error,mean_squared_error 
import numpy as np 
# real score
real_scores=[90,60,80,100]
# model guess
predicted_scores =[85,70,70,95]
mae=mean_absolute_error(real_scores,predicted_scores)
print("mean_absolute_error")
print(mae)
mse=mean_squared_error(real_scores,predicted_scores)
print("mean_squared_error ")
print(mse)
rmse=np.sqrt(mse)
print("root mean squared error:")
print(rmse)


# 1. MAE (Mean Absolute Error)

# Use jab: Tumhe sirf average galti chahiye, aur bade errors ko extra importance nahi dena.

# Kaha use hota hai:

# Delivery time prediction (galti ±5 minute ya ±10 minute ka average important hai).

# House price prediction (har galti ko equal treat karna hai).

# Robust systems jaha outliers (bahut ajeeb data) ignore karna hai.

# 👉 MAE easy to interpret hai: “hamari prediction average itni unit galat hai.”

# 2. MSE (Mean Squared Error)

# Use jab: Tumhe chahiye ki system bade errors ko heavily punish kare.
# Par iska nuksan → unit change ho jata hai.
# Kaha use hota hai:

# Machine Learning training (Linear Regression, Neural Networks me cost function mostly MSE hota hai).

# Finance risk analysis (chhoti galti chalti hai, lekin badi galti bahut khatarnaak hai).

# Medical field me disease prediction (ek patient ka zyada galat prediction avoid karna chahte ho).

# 👉 MSE training ke liye best hai kyunki derivatives smooth hote hain.

# 3. RMSE (Root Mean Squared Error)

# Use jab: Tumhe chahiye ki error ka value same unit me aaye jaise actual data.

# Kaha use hota hai:

# Weather forecasting (temperature ka error “degree Celsius” me chahiye).

# Traffic prediction (average error kitne cars ka hai).

# Energy consumption forecasting (kWh unit me galti).

# 👉 RMSE report karne ke liye best hai, samajhna easy hota hai.
# Matlab RMSE ek balance hai → interpret karne me easy (jaise MAE) + bade errors ko importance deta hai (jaise MSE).