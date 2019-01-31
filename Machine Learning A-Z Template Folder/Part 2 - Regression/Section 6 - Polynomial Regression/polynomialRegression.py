# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:48:16 2018

@author: Maria
"""

#Polynomial Linear Regression: y = b0 + b1x1 + b2x1^2 + ...+ bnx1^n. Called linear ecause of the coefficient
#Polygonial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Importing the dataset
dataset = pd.read_csv('C:/Users/Maria/Documents/MyCodes/Machine Learning Course/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #X is considered as a matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

"""small set of data, we won't split the data.
We don't need scaling here"""

#fitting linear regression as a reference
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regression model
poly_reg = PolynomialFeatures(degree = 4) #transforms X into a new matrix of features X_poly
X_poly = poly_reg.fit_transform(X) #created another matrix with squeared etc and included column of ones
#create another linear regresson το include fit into a multiple multilinear model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) #the second regresson object to the X_poly matrix contαing two indepented variables

#Visualizing the linear Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Visualizing the polynomial Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#We have straight lines between each level, so we create 
X_grid = np.arange(min(X), max(X), 0.1) #for a smooth line
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)#6.5 is the level that we want to predict

#Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5)) #close to the results