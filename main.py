#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import minimize


#Data Preprocessing
cric_data = pd.read_csv('../Data/04_cricket_1999to2011.csv')
train = preprocess(cric_data)


#Objective Function for minimization
def objective_function(Params, train):
    squared_loss = 0
    Total_data_points = 0
    for i in range(10):
        wk = train[train['Wickets.in.Hand'] == 10-i].reset_index()
        u = 50 - wk['Over']
        Z_actual = wk['Innings.Total.Runs'] - wk['Total.Runs']
        Z_pred = Params[i] * (1 - np.exp((-Params[10]*u)/Params[i]))   #params[0] to params[9] = Z0(10) to Z0(1) and params[10 = L]
        squared_loss = squared_loss + np.sum((Z_pred- Z_actual)**2)
        Total_data_points = Total_data_points + np.size(u)
    
    return squared_loss/Total_data_points


#Initialization
Params0 = np.array([275,240,218,180,150,111,95,60,35,15,5], dtype = np.float64)


#Non-linear Regression (Optimization)
result = minimize(objective_function, Params0, args = (train), method = 'CG')
print('#################################################################\nOptimizer Output')
print(result)


#Plots
plt.figure(dpi=200)
for i in range(10):
    overs = np.arange(0, 51, dtype = int)
    Z   = result.x[i] * (1 - np.exp((-result.x[10]*overs)/result.x[i]))
    plt.plot(overs, Z)
    plt.grid('major')
plt.xlabel('Overs Remaining')
plt.ylabel('Average Runs Obtainable')
plt.title('Run Production Functions')
plt.legend(['10','9','8','7','6','5','4','3','2','1'], loc ="upper left", prop={'size': 7})
plt.show()
