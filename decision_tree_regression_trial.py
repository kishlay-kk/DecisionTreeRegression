# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 19:15:35 2018

@author: kishl
"""
# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the Dataset
dataset=pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
exp = 6.5

# Creating Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)
 
# Predicting the value
y_pred = regressor.predict(exp)

# Visualising the Decision Tree Regression
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
