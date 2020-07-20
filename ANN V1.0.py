#V.1 Artificial Neural Network
#http://iamtrask.github.io/2015/07/12/basic-python-network/


#Importing Library 
import numpy as np 

#Sigmoid Function  
def nonlin(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    
    return 1 / (1 + np.exp(-x))

#Input dataset values
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ] )

#Output Dataset
y = np.array( [[0,0,1,1]] ).T  

#Seed random nos. to make calculation deterministic (good practise)
np.random.seed(1)

#Initialize weights randomly with mean 0 (Essentially this is the neural network; l0 and l1 are transient values stored in syn0)
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    
    #forward propagation 
    l0 = X
    l1 = nonlin(np.dot(l0,syn0)) #Prediction step
    
    #how much did we miss 
    l1_error = y - l1 
    
    #multiply how much we missed * slope of sigmoid at the values in l1 
    l1_delta = l1_error * nonlin(l1,True) # Error  weighted derivative - REDUCING THE ERROR OF HIGH CONFIDENCE PREDICTIONS 
    
    #Update weights = input_value(l0) * l1_delta 
    syn0 += np.dot(l0.T,l1_delta)
    
print ("Output after Training ")
print (l1)

    
    