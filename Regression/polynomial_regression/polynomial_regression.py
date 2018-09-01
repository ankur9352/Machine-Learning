
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv("Position_Salaries.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor=PolynomialFeatures(degree=4)
x_poly=polynomial_regressor.fit_transform(x)
polynomial_regressor.fit(x_poly,y)
regressor2=LinearRegression()
regressor2.fit(x_poly,y)
print(x_poly)
# linear egression between x and y
plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")
plt.show()
# linear regression between x_poly and y
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor2.predict(polynomial_regressor.fit_transform(x_grid)),color="blue")

plt.show()
