import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv("Position_Salaries.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.show()
