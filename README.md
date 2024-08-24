# Nexford_BAN_PCA


## README

## ANDERSON CANCER CENTER DATA ANALYSIS

This is a project is aimed at developing a model to address the growing number of referrals at the Anderson Cancer Center.This model's first step is to identify essential variables for securing donor funding. And the most suitable technique for this is the Principal Component Analysis (PCA).



## INSTRUCTIONS

We will perform the following tasks:

1.	PCA Implementation:

  o	Utilize PCA to demonstrate how essential variables can be acquired from the cancer dataset available from sklearn.datasets.

2.	Dimensionality Reduction:

  o	Reduce the dataset into 2 PCA components for the project.

3.	Bonus Point (Optional):

  o	Implement logistic regression for prediction.


## INSTALLATION

Install sklearn in Python
```bash
pip install scikit-learn
```

## Running Python


#Steps To Perform Principal Component Analysis (PCA) in Python:

Import all libraries
```bash
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

Load the dataset from built-in scikit-learn
```bash
from sklearn.datasets import load_breast_cancer
```

Standardizie the data
```bash
scaler = StandardScaler()
scaled_data =scaler.fit_transform(df)
```

Create an instance of the PCA classroom form scikit-learn
```bash
pca = PCA(n_components=2)
```

Use the fitted PCA model to transform your data. 
```bash
principal_components = pca.fit_transform(scaled_data)
```

Create a new DataFrame to store two principal components
```bash
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
```

Plot the principal components to visualize the data in the reduced-dimensional space
```bash
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=data.target)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cancer Center')
plt.show()
```

Display the first few rows and columns in the Dataframe
```bash
print(df.head())
```

#Steps To Implement Logistic Regression For Prediction:

Import all libraries
```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

Load the dataset from built-in scikit-learn
```bash
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

Standardizie the data
```bash
scaler = StandardScaler()
scaled_data =scaler.fit_transform(df)
```

Split your dataset into features (X) and target (y).
```bash
X = df.drop('target', axis=1)
y = df['target']
```

Divide the data into training and testing sets. 
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Create an instance nd fit the model to the training using fit
```bash
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

Use the trained model to make predictions on the testing set.
```bash
y_pred = model.predict(X_test)
```

Evaluate the model (Calculate accuracy, confusion matrix and classification report)
```bash
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)
```

#Note: 
Visulaiztions and results will be output in the console. 
## Lessons Learned

This project which is aimed at developing a model for the Anderson Cancer Center to adress the growing number of referrals, and demonstrates how the Principal Component Analysis (PCA) is been used to identify variables for securing donor funding in the breast cancer dataset and perform dimensionality reduction.

First, scikit was installed into python which will give us an in-built dataframe for the breast cancer. Then we import all the libraries necessary for the for data manipulation, numerical operations, and visualization i.e pandas, numpy, sklearn and matplotlib.

Secondly, we laod the dataset from scikit-learn and then data so that each feature has a mean of zero and a standard deviation of one. Next, we create an instance of two PCA class from scikit-learn.

Thirdly, we transform the data using the PCA fitted model that ccaptures the most variance in the data. Next, we create a a new dataframe to store the principal components.

Lastly, we plot the principal components to visualize the data in the reduced-dimensional space and it gives a scatter plot of the first two principal components.Then when we implemented the logistics regression model to prediction cancer dianosis using the reduced PCA components. 
## Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
