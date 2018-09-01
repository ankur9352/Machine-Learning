
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
data=pd.read_csv("50_Startups.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
ss_x=StandardScaler()
# x_train=ss_x.fit_transform(x_train)
# x_test=ss_x.fit_transform(x_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
rmsc=np.sqrt(mean_squared_error(y_test,y_predict))
print(rmsc)
# print(np.mean(y_predict==y_test))


