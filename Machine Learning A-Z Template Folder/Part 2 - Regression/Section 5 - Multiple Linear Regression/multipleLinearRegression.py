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
#1. All-in --> cases: Prior Knowledge; OR
#             You have to; OR, for example according to framework of the company
#             Preparing for Backward Elimination

#2. Backward Elimination -->
#       step 1. :Select a significance level to stay in the model, SL = 0.05
#      step 2. : fit the model to possible predictors
#      step 3. : Consider the predictor with the highest p-value. If P>SL, go to step 4, otherwise go to FIN
#      step 4. : Remove the predictor
#      step 5. : Fit model without this variable (go to 3)
#Fin => model ready

#3. Forward Elimination -->
#        step 1. :  Select a significance level to stay in the model, SL = 0.05
#      step 2. : Fit all simple regression models y~xn select the one with the lowesr P-value
#      step 3. : keep this variable and fill all possible with one extra predictor added to the ones you already have
#      step 4. : consider the predictor with the lowest P-value. if P<SL, go to step 3, otherwise go to FIN(keep the previous model)

#4. Bidirectional Elimination -->
#       step 1. : step 1: select a significance level to enter to stay in the model e.g.: SLENTER = 0.05, SLSTAY = 0.05
#      step 2. : Perform the next step of Forward Selection (new variables must have: P<SLENTED to enter)
#      step 3. : Perform ALL steps of Backward Elimination (old variables must have P<SLSTAY to stay)
#      step 4. : No new variables can enter and no old variables can exit
#=> FIN: your model is ready

#5.Score Comparison -->
#       step 1. : Select a criterion of goodness of fit (e.g. Akaike criterion)
#      step 2. : Construct All Possible Regression Models: 2^N -1 total combinations
#      step 3. : Select the one with best criterion
# FIN => Your model is ready. Example: 10 columns means 1,023 models



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Maria/Documents/MyCodes/Machine Learning Course/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray() #fit transform to dataset

#avoiding the dummy variable trap
X = X[:, 1:] #I remove the first column. python library for linear regression already takes care of that

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#fitting multiple linear regression to trining test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#this plot would be 5-dimentional so we won't plot it here

#With this model we used all variables. We could use less with:
#Backward Elimination
import statsmodels.formula.api as sm
#the constant of the multiple regression equation actually is multiplied with x0=1. I have to include that to the code because the library dosn't include it
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)#we add a column of ones. we also have to make the added column integers. note that this adds the X to ones, not the other way arround

#we will create a new matrix of features that will be the optimal matrix of features
X_opt = X[:, [0,1,2,3,4,5]]#we will initialize the features and we will remove them ine by one (the ones that are statistically more useeless)
#1st - 2nd step: select significance level and fit
#create a new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#regresson object from the OSL class
#step 3: consider predictor with highest p-value
regressor_OLS.summary()#this shoes the p values. so we remove the predictor with the highest p-value

#we do the same without the third column
X_opt = X[:, [0,1,3,4,5]]#we will initialize the features and we will remove them ine by one (the ones that are statistically more useeless)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#regresson object from the OSL class
regressor_OLS.summary()#this shoes the p values. so we remove the predictor with the highest p-value

#we do the same
X_opt = X[:, [0,3,4,5]]#we will initialize the features and we will remove them ine by one (the ones that are statistically more useeless)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#regresson object from the OSL class
regressor_OLS.summary()#this shoes the p values. so we remove the predictor with the highest p-value

#we do the same
X_opt = X[:, [0,3,5]]#we will initialize the features and we will remove them ine by one (the ones that are statistically more useeless)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#regresson object from the OSL class
regressor_OLS.summary()#this shoes the p values. so we remove the predictor with the highest p-value

#we do the same
X_opt = X[:, [0,3]]#we will initialize the features and we will remove them ine by one (the ones that are statistically more useeless)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#regresson object from the OSL class
regressor_OLS.summary()#this shoes the p values. so we remove the predictor with the highest p-value

#This is our model!


#a little les manualy solution would be:
'''import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)'''

# with both p-values and adjusted R squeared, just in case p-value~0,05, so I have one more criteria to choose I if I will eliminate the variable or not

'''import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)'''





