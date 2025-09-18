import pandas as pd 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

# sample data
data={
    'name':["riya","aman","shashank","krishna",'radha',"rahul"],
    'age':[20,30,40,22,38,25],
    'spending':[100,200,300,110,290,130]
}
df =pd.DataFrame(data)

# clustering
X=df[['age','spending']]
model =KMeans(n_clusters=2,random_state=42,n_init=10)
df['Group']=model.fit_predict(X)
plt.figure(figsize=(6,5))
for group in df['Group'].unique():
    # this code menas pandas [[T,F,T,F,F,T,T]] ISE MAI SE TRUE WALIS ROWS KO UTHA LEGA
    group_data=df[df['Group']==group]
    plt.scatter(group_data['age'],group_data['spending'],label=f'Group{group}')
plt.xlabel('age')    
plt.ylabel('spending score')
plt.title('Customer segment (k-Means)')
plt.legend()
plt.grid(True)
plt.show()
print(df)
# plt.scatter(group_data['age'], group_data['spending']) → graph banane ke liye (do alag lists pass karni hoti hain).


