
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv("Position_Salaries.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values
y=y.reshape((len(y),1))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel="rbf")
regressor.fit(x,y)

y_pred = regressor.predict(sc_X.fit_transform(np.array((6.5))))

plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")
plt.show()

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.show()
