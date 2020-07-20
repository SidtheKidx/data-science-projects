#Upper Confidence Bound 

#Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#Loading the dataset 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

"""
#Implementing UCB 
import math 
N= 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selection[i] > 0):
        
            avg_reward = sums_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selection[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    numbers_of_selection[ad] = numbers_of_selection[ad] + 1       
    reward = dataset.values[n, ad]  
    sums_of_rewards[ad] + reward 
    total_reward = total_reward + reward 
"""

# UCB algorithm
import math
(N,d)=(dataset.shape[0],dataset.shape[1]) #get row and col values
(num_of_selections,sum_of_rewards) = ([0] * d,[0] * d) #initialize
(avg_reward,ucb) = ([0] * d,[0] * d) #initialize
ad_selected=[] #initialize
ad=0 #initialize

for n in range(0,N): #read thru every row
    
    if n < d: #have we tried all ads atlest once
        ad = n #try that ad
    else:
        ad = ucb.index(max(ucb)) #else get ad with max upper confidence bound
    
    ad_selected.append(ad)   #add the ad selected 
    sum_of_rewards[ad] += dataset.values[n,ad] #update the reward for selected ad
    num_of_selections[ad] += 1 #update number of selection for selected ad
    avg_reward[ad] = sum_of_rewards[ad]/num_of_selections[ad] #update avg reward for selected ad
    delta = math.sqrt(3/2*math.log(n+1)/num_of_selections[ad]) #delta for selected ad
    ucb[ad]= avg_reward[ad]+delta #update upper confidence bound for selected ad
total_rewards = sum(sum_of_rewards) #calcualte total rewards earned


#Visualising
import seaborn as sns
sns.set()

plt.hist(ad_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('No. of each ad was selected')
plt.show()
