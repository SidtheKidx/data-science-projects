#Simple Linear Regression 

#Importing libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Importing Dataset
dataset= pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitiing the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

"""
#Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 

#Predicting test set results 
y_pred = regressor.predict(X_test)

#Visualising the training set results 
plt.scatter(X_train, Y_train, color = 'red' )
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('SALARY vs. EXPERIENCE (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



#Visualising the test set results 
plt.scatter(X_test, Y_test, color = 'red' )
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('SALARY vs. EXPERIENCE (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

