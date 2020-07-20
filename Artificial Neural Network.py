#ANN 

#Importing the libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#importing the dataset 
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[ :, 3:13 ].values 
y = dataset.iloc[ :, 13].values 

"""# encoding categorical data
#to avoid dummy variable n-1 columns are made 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder="passthrough")
ct_country_gender = np.array(ct.fit_transform(X)[:, [1,2,3]], dtype=np.float)
X = np.hstack((ct_country_gender[:, :2], dataset.iloc[:, 3:4].values, ct_country_gender[:, [2]], dataset.iloc[:, 6:-1].values))
"""

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#To eliminate dummy variable trap 
X = X[:, 1:]

#splitting the dataset 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ANN creation 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 

#Initialising an ANN 
classifier = Sequential()

#Adding he input layer and first hidden layer - Avg of [nodes in I/p + nodes in O/p layer]
classifier.add(Dense(units= 6, input_dim = 11, kernel_initializer='uniform', activation = 'relu'))

#Adding the second hidden layer 
classifier.add(Dense(units= 6, kernel_initializer='uniform', activation = 'relu'))

#Adding the output layer 
classifier.add(Dense(units= 1, kernel_initializer='uniform', activation = 'sigmoid'))

#Compiling (Stochastic Gradient Descent) the ANN 
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fitting the ANN with Training set 
classifier.fit(X_train, y_train, batch_size= 10, epochs= 100)

#Making the predictions and evaluating the model 
y_pred = classifier.predict(X_test)
y_pred= (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)