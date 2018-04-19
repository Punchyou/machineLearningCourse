# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:04:13 2018

@author: Maria
"""

#Data processing

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')

#in ML we have to determine the matrix features and the dependened vasiables vector

#we make a matrix of the variables  ()features)from the dataset (3 firsts columns of dataset)
X = dataset.iloc[:, :-1].values #: for taking all the lines,  :-01 for taking all the columns exept tha last one

#create the depented variabe vector
y=dataset.iloc[:, 3].values #3for taking the last column

'''#in case of missing data, we fill the empty space with the mean of all the oservations of the column
from sklearn.preprocessing import Imputer #we use skickit learn imputer class that takes care of missing data

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #clt + i to help. 0 for columns.
imputer = imputer.fit(X[:,1:3]) #we take only the columns that have missing data. So, : for all lines and 1:3 for taking 2nd and 3rd columns (1 and 2 for python. 3 is not included)
X[:, 1:3] = imputer.transform(X[:,1:3]) #we call the function transform to aply the tranform we want

#we have categorical variables if columns contain categories (for example not numbers)
#here we have countries and purchased
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #now i encoded the country column

#the encoding is 0,1,2. python now thinks that 1 country is now greater than 0 country. we have to prevent that from ML algorithm
#so instead of one column, we will create 3 (one for each country)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray() #fit transform to dataset


#no need to use OneHotEncoder for the second categorical column (purchased)
#we will only label encoder
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)'''


#need for spliting to trainig (on which we built the ML algorithm) set and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#20% of the data will be uset to train the system. since we input the test size there's no need to do so for the train set. (test+train = 1 anyway). random state = <specific number> for having the same results as the course

#overfitting: the system understands the correlations between features and dependent variables, but not the logic behund it. This way we may have wrong results

'''#feature scaling: some variable columns don't have scale. many ML algorithms are based in euclidean distance. we may use standartisation or normalisation. It's anyway quicker, and good to apply it in every dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #I have to fit the training set and then transform it
X_test = sc_X.transform(X_test) #we don't fit the test set cause we already fitted the training set'''

#preprocessing ready!
