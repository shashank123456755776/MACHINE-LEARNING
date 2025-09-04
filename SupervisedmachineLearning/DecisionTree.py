from sklearn.tree import DecisionTreeClassifier 
X=  [[7,2],#apple
    [8,3],#apple
    [9,8],#orange
    [18,9],#orange
]
y=[0,0,1,1]
model = DecisionTreeClassifier()
model.fit(X,y)
size=float(input("enter size in cm:"))
shade=float(input("enter color shade (1-10):"))
result=model.predict([[size,shade]])[0]
if result ==0:
    print("This is likely to be Apple")
else:
    print("This is likely to be Orange")
        
#Underfitting --deeply focus on given example if yoy give another exapmples to predict it fails
#Overfititng--menas he will consider evrything same like cat exapmle
#Goodfitting-- hmm models ko particular chijj baatate hai       