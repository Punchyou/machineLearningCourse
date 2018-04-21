# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:48:16 2018

@author: Maria
"""

#Polynomial Linear Regression: y = b0 + b1x1 + b2x1^2 + ...+ bnx1^n. Called linear ecause of the coefficient
#Polygonial Regression


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #X is considered as a matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#if we have small set of data we don't split the data. In onder to make the more acurate results we have to train out system with maximum training set availiable
# We don't need scaling here, library already takes care of that

#fitting linear regression as a reference
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #transformer tool tha transforms X into a new matrix of features X_poly. we can try 2,3,4 and plot it. whatever works better
X_poly = poly_reg.fit_transform(X) #check the X_poly matrix. we created another matrix with squeared etc. also included column of ones
#we have to include this fit into a multiple multilinear model. We have to create another linear regresson
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) #we fit the second regresson object to the X_poly matrix conteing two indepented variables

#Visualizing the linear Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Visualizing the polynomial Regression results
#Visualizing the linear Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#We have straight lines between each level
#so we create 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()





