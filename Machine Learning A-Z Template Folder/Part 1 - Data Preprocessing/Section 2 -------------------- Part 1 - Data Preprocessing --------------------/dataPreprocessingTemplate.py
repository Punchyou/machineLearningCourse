# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:04:13 2018

@author: Maria
"""

#Data processing

import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #: for taking all the lines,  :-01 for taking all the columns exept tha last one

#create the depented variabe vector
y=dataset.iloc[:, 3].values 

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3]) #we take only the columns that have missing data
X[:, 1:3] = imputer.transform(X[:,1:3])

'''
#Encoding categorical variables (countries and purchased)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#the 0, 1, 2 gives different value t each country
#so, use onehotencoder to crete 3 lists instead of three(one for each country)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder() #no need to use OneHotEncoder
y = labelencoder_X.fit_transform(y)
'''

#spliting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #fit the training set and then transform it
X_test = sc_X.transform(X_test) #don't need to fit the test (we already fitted the training set)

#preprocessing ready!"""