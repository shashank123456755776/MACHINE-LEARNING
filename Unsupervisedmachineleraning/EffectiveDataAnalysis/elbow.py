import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data
data = {
    'age':[20,30,40,22,38,25,45,50,60,55],
    'spending':[100,200,300,110,290,130,400,410,500,450]
}
df = pd.DataFrame(data)
X = df[['age','spending']]

# Step 1: Calculate WCSS for k = 1 to 10
wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)  # inertia_ = WCSS

# Step 2: Print WCSS values
print("WCSS for different k values:")
for i, val in enumerate(wcss, start=1):
    print(f"k={i}: WCSS={val:.2f}")

# Step 3: Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()
# WCSS hume batata hai ki clusters kitne “tight” hain; jitna low WCSS, utna points cluster ke beech closely packed hain.
# kabhi kabhi k ni pata hota hai to hamm elbow methods use karte hai 
# Within Cluster Sum of Squares calculated by machine to find k 

# 1️⃣ WCSS kam = clusters tight

# Agar aap k bahut bada rakh do (jaise 10 clusters for 10 points),

# Har point ka apna cluster hoga → WCSS 0 ke paas ho jayega

# Matlab points bahut tight hain… lekin ye useful clustering nahi hai

# 2️⃣ Optimal k = Elbow Point

# Elbow method me aap **wo point choose karte ho jahan WCSS kam hona slow ho jaye

# Matlab:

# Pehle k badhane par WCSS kaafi kam hota hai → clusters tighter hote hain

# Ek point ke baad WCSS kam hona slow → cluster enough tight hai → aur zyada clusters add karna unnecessary hai

# ✅ Rule of thumb:

# Na too few clusters (WCSS high → clusters loose)

# Na too many clusters (WCSS very low → overfitting)

# Just elbow point → clusters tight + meaningful




# Agar sirf 1 cluster hai (k = 1):

# 1️⃣ Situation

# Sab points ek hi cluster me hain

# Cluster ka centroid = sab points ka average

# 2️⃣ WCSS

# WCSS = har point aur centroid ke beech distance squared ka sum

# Kyunki points alag-alag hain → centroid se kaafi door ho sakte hain

# Isliye WCSS HIGH hoga

# 3️⃣ Intuition
# k	WCSS
# 1	High (sab points ek cluster me, loose)
# 2	Lower (points 2 clusters me, tighter)
# 3	Even lower (points aur tightly clustered)

# → Jaise-jaise k badhega, points apne cluster ke pass aayenge → WCSS ghatega

# 💡 Summary:

# 1 cluster → WCSS high

# Zyada clusters → WCSS low

# Optimal clusters = elbow point → WCSS enough low + clusters meaningful