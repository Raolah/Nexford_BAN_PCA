#Implementation of PCA using the cancer dataset from sklearn.datasets
#Import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Load dataset from sklearn.dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#Standardize the data
scaler = StandardScaler()
scaled_data =scaler.fit_transform(df)

#Initialize and fit PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

#Create a dataframe with two princpal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

#Dimensionality Reduction
#Visualization of the princical components
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=data.target)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cancer Center')
plt.show()

print(df.head())

