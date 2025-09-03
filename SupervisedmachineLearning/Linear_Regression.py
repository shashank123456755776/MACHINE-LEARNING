# Linear Regression means models used the numbers based on numbers gives answers 
# First bracket → dataset (list of samples)
# Second bracket → features inside each sample
# Features →   Age    Height   Hours studied
#              │       │          │
#              ▼       ▼          ▼
# Samples ─► [ 18 ,   170 ,       3 ]   ← Sample 1
#           [ 20 ,   165 ,       5 ]   ← Sample 2
#           [ 19 ,   172 ,       2 ]   ← Sample 3

from sklearn.linear_model import LinearRegression 
model=LinearRegression()
X=[[1],[2],[3],[4],[5]]
y=[48,58,65,75,90]
model.fit(X,y)
hours=float(input("Enter how many hours you studied="))
predicted_marks =model.predict([[hours]])
print(f"Based on your Hours {hours} you may score {predicted_marks}")