# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:21:47 2021

@author: bharghava
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from collections import Counter
from IPython.core.display import display, HTML
sns.set_style('darkgrid')
from sklearn.datasets import load_boston
boston_dataset = load_boston()
dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
## TO sSHOW FIRST FIVE ROWS OF THE DATA
dataset.head()
## SET THE TARGET DATA
dataset['MEDV'] = boston_dataset.target
dataset.head()

##DATAPREPROCESSING 
##STEP-1: FIND THE MISSING VALUE
dataset.isnull().sum()# HRE THERE IS NO NULL VALUE
##STEP-2 # RESHAPE THE DATASET
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values.reshape(-1,1)

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)
## TO SEE THE SHAPE OF TRAIN AND TEST DATASETS
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)
##Visualizing Data ## EDA
corr = dataset.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns);
#show plot
plt.show()
import seaborn as sns
sns.pairplot(dataset)
plt.show()
## CEREATE MODEL
##Polynomial Regression - 2nd degree 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
###FITTING THE MODEL
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
###USING LINEAR REGRESSION TO FIT THE POLYNOMIAL REG
from sklearn.linear_model import LinearRegression
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_poly, y_train)
print(regressor_poly2)
## FINDING ACCURACY
from sklearn.metrics import r2_score
# Predicting Cross Validation Score the Test set results
cv_poly2 = cross_val_score(estimator = regressor_poly2, X = X_train, y = y_train, cv = 10)
print('crossvalidation', cv_poly2)

##PREDICTING 
# Predicting R2 Score the Train set results
y_pred_poly2_train = regressor_poly2.predict(poly_reg.fit_transform(X_train))
r2_score_poly2_train = r2_score(y_train, y_pred_poly2_train)
print(y_pred_poly2_train)
print(r2_score_poly2_train)

# Predicting R2 Score the Test set results
y_pred_poly2_test = regressor_poly2.predict(poly_reg.fit_transform(X_test))
r2_score_poly2_test = r2_score(y_test, y_pred_poly2_test)
print(y_pred_poly2_test)
print(r2_score_poly2_test)

# Predicting RMSE the Test set results
rmse_poly2 = (np.sqrt(mean_squared_error(y_test, y_pred_poly2_test)))
print('CV: ', cv_poly2.mean())
print('R2_score (train): ', r2_score_poly2_train)
print('R2_score (test): ', r2_score_poly2_test)
print("RMSE: ", rmse_poly2)






































