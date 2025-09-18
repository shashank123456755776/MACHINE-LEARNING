# 1️⃣ Full Form

# PCA = Principal Component Analysis

# 2️⃣ PCA ka Purpose

# PCA ek dimensionality reduction technique hai:

# Matlab agar aapke data me bahut saare features (columns) hain → PCA unko kam dimensions me convert karta hai

# Goal: important information preserve karna aur redundant info hatana

# 3️⃣ Kaise kaam karta hai (intuition)

# Imagine karo aapke paas 2D data hai (x, y):

# Points ka spread diagonal me hai

# PCA nikalta hai ek new axis jo data ke maximum variance ko capture kare

# 2D → 1D me reduce karna possible hai without losing much info

# Step by step intuition:

# Data ke mean ke around center banao

# Covariance matrix calculate karo → features ke relationships pata chale

# Eigenvectors & Eigenvalues nikalna → ye direction (principal components) aur importance (variance) batata hai

# Data ko principal components me project karo → reduced dimension



# PCA --PRINCIPLE COMPONENTS ANALYSIS 
#  Why use PCA?

# Data visualization → high-dimensional data 2D/3D me plot kar sakte ho

# Noise reduction → less important features remove kar sakte ho

# Faster computation → fewer features = machine learning models faster


# Standard Scaler ka Estammal tab karte hai jab hame data ko scale karna hota hai 
# pca karne se pahle scaling karna important hai ku ki large data ke liye ager pca lageyege to importnat data loss ho jayega
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  

data = {
    "Age":[25,30,35,40,45,50],
    "Income":[30000,40000,50000,60000,70000,80000],
    "spending":[70,60,50,40,30,20],
    "savings":[1000,5000,8000,10000,15000,20000]
}

df = pd.DataFrame(data)

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
print("pca result :",pca_result)
pca_df = pd.DataFrame(pca_result, columns=["PCA1","PCA2"])
print(pca_df)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:")
print(np.round(explained_variance*100,2))

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(pca_df["PCA1"], pca_df["PCA2"], color="black")
plt.title("PCA Projection")
plt.xlabel("PCA1 Main Pattern")
plt.ylabel("PCA2 Minor Pattern")
plt.grid(True)
plt.show()
print("new data with 2 features pca1 and pca2")
print(pca_df)
# pca ka center 0,0 hota hai
