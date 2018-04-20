# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:34:05 2018

@author: Maria
"""
#multiple linear regression

#it uses y = b0 _ b1*x1 + b2*x2 + b3*x3 + (b4*D), the larenthesis contains the part of the equation for categorical data
#p-value is used, check here for more: https://www.mathbootcamps.com/what-is-a-p-value/
#Linear regression have assumptions (need to check if these are true): 1. Linearity, 2. Homoscedasticity, 3. Multivariate normality, 4.Undependence of errors, 5. Lack of multicollinearity. we won't need these here
#we use dummy variables instead of categorial variables
#if n equals the number of dummy variables, we use n-1 variables in the part of the equation that is about the dummy variables
# Data Preprocessing Template


#5 methids of building models: 1. All in, 2. Backward Elimination, 3. Forward Selection, 4. Bidirectional Elimination, 5. Score Comparison
# 2-4 ---> stepwise regression

#Step by step
#1.--> cases: Prior Knowledge; OR
#             You have to; OR, for example according to framework of the company
#             Preparing for Backward Elimination

#2.--> step 1. :Select a significance level to stay in the model, SL = 0.05
#      step 2. : fit the model to possible predictors
#      step 3. : Consider the predictor with the highest p-value. If P>SL, go to step 4, otherwise go to FIN
#      step 4. : Remove the predictor
#      step 5. : Fit model without this variable (go to 3)
#Fin => model ready

#3. --> step 1. :  Select a significance level to stay in the model, SL = 0.05
#      step 2. : Fit all simple regression models y~xn select the one with the lowesr P-value
#      step 3. : keep this variable and fill all possible with one extra predictor added to the ones you already have
#      step 4. : consider the predictor with the lowest P-value. if P<SL, go to step 3, otherwise go to FIN(keep the previous model)

#4.--> step 1. : step 1: select a significance level to enter to stay in the model e.g.: SLENTER = 0.05, SLSTAY = 0.05
#      step 2. : Perform the next step of Forward Selection (new variables must have: P<SLENTED to enter)
#      step 3. : Perform ALL steps of Backward Elimination (old variables must have P<SLSTAY to stay)
#      step 4. : No new variables can enter and no old variables can exit
#=> FIN: your model is ready

#5.--> step 1. : Select a criterion of goodness of fit (e.g. Akaike criterion)
#      step 2. : Construct All Possible Regression Models: 2^N -1 total combinations
#      step 3. : Select the one with best criterion
# FIN => Your model is ready. Example: 10 columns means 1,023 models


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#we have categorical variables--> label encoder and onehat encoder
#encoding categorical data and
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #now i encoded the country column
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray() #fit transform to dataset


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

