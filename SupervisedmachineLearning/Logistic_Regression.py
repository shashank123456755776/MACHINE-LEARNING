# ek classification model jiska use karte hai hmm binary output predict karne ke liye using input features
# 
    
    # KNN algorithm works based on neraest data exapmle mail spam or not
from sklearn.neighbors import KNeighborsClassifier
# eucludean distance
X=[
    [180,7],
    [280,7.5],
    [250,8],
    [300,8.5],
    [330,9],
    [360,9.5]
]    
y=[0,0,0,1,1,1]
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
weight=float(input('enter weight:'))
size=float(input('enter size:'))
result=model.predict([[weight,size]]) [0]
if result==1:
    print("This is Likely to be apple")
else:
    print("This is Likely to be Orange")    
    
    