# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:56:55 2018

@author: Maria
"""

#Polynomial Regression Template

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #X is considered as a matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set. We have to uncomment that for a bigger dataset
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling for big dataset
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting the Regression Model to the dataset
#Create your regressor here


#Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

#Visualizing the Polynomial Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#for a smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()


