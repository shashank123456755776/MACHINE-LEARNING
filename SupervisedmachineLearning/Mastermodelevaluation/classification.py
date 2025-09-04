#accuracy-correct prediction/total prediction *100
#precision-ager error avoid karna ho
#recall --Real data ko match karnan ho ager
#f1 score--data imbalance --it should be balance
#sklearn.matrics not use to manually calculate the accuracy,recall to match the real data In python it happen Automatically 

from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score
# what actually happen
y_True=[1,0,1,1,0,1,0]
# what actually Gussesd
y_Predict=[1,0,1,0,0,1,1]
#evaluation
print('Accuracy',accuracy_score(y_True,y_Predict))
print('Precision',precision_score(y_True,y_Predict))#postive prediction example 10 mai 6 sahi kiye to 6/10 *100
print('recall',recall_score(y_True,y_Predict))#real prediction
print('F1 Score',f1_score(y_True,y_Predict))#f1 score
