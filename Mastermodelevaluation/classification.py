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
#data imballanced mai accuracy kaam ni karegi

#TP MACHINE SAY YOU WILL BE PASS BUT ACTUALLY  PASS
#TN MACHINE SAY YOU WILL BE FAIL SO ACTUALLY FAIL
#FP MACHINE SAY YOU WILL BE PASS BUT ACTUALLY FAIL
#FN MACHINE SAY YOU WILL BE FAIL BUT ACTUALLY PASS
# Precision → "Jo machine ne pakda, usme kitne waaqai chor the?"
# Recall → "Jo waaqai chor the, unme se machine ne kitne pakde?"
# TP → Sahi chor pakda

# TN → Sahi innocent bachaya

# FP → Innocent ko chor bola

# FN → Chor ko chhod diya
# YEHA JO TP TN FP FN AA REHE HAI YE MACHINE PREDICTION AUR REAL DATA KO CaMPARE KAR KE AA REHA HAIN 
# precision=TP/TP+FP 15/25
# RECALL=TP/TP+FN 15/20 yeha 20 actual data hai ,15 compare wala hai and 25 prediction hai
# FN=Actual Positive−TP

# ⚖️ Easy way to think (without formula):

# Reality list banao → Kaun sach me cheater hai (20 log).

# Machine list banao → Machine ne kaun pakde (25 log).

# Compare karo:

# Jo dono lists me hain = TP

# Jo reality me hain par machine ki list me nahi aaye = FN

# 👉 Matlab FN wahi hain jo ground truth me the, par machine ki prediction se bahr reh gaye.

# 🎯 Example with names (samajhne ke liye)

# Actual cheaters (20) = {A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T}

# Machine ne pakde (25) = {A, B, C, D, E, F, G, H, I, J, U, V, W, X, Y, Z, P, Q, R, S, M, N, O, K, L}

# 🔍 Ab dekhte hain:

# Common part (dono lists me) = TP = 15

# Actual list me jo bache aur machine ne miss kar diye = FN = 5

# ✅ Core concept:

# FN ka matlab = reality me chor tha, lekin machine fail ho gayi pakadne me.

# Isliye FN “missed positives” hote hain.