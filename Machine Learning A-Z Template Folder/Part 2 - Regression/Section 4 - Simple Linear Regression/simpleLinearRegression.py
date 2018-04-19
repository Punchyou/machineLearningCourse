# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:51:42 2018

@author: Maria
"""
#Simple Linear Regression: simple linear Regression. Use oridinary least squears: sum(y -yi)^2 and fine the minimum, for finding the best line.

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3 , random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fotting simple linear regression to the training test with a lib
from sklearn.linear_model import LinearRegression
#create an object of the linear regression class
regressor = LinearRegression() #every parameter is optional here so we don't use any
regressor.fit(X_train, y_train)

#we made the regressor machine that learns the correlations between X and y from training data

#Predicting the Test set Results
y_pred = regressor.predict(X_test)#VECTOR of predictions

#Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red') #plot training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#plot training set with linear regression
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red') #plot training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#no need to change to change train to test set. regressor is alreadt trained
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


