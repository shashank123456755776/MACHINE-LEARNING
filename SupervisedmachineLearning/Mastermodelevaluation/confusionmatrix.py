from sklearn.metrics import confusion_matrix
#actual true 
y_true=[1,0,1,1,0,1,0,0,1,0]
#prediction
y_pred =[1,0,1,0,0,1,1,0,1,0]
cm=confusion_matrix(y_true,y_pred)
print("confusion matrix:")
print(cm)
# Confusion matrix very used  in fraud detection and crime detection
# [[TN FP][FN][TP]]
