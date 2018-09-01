
import pandas as pd 

data=pd.read_csv("Salary_Data.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict=regressor.predict(x_test)

import matplotlib.pyplot as plt 
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test,y_predict, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
