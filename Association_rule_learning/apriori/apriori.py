
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
data=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions=[]
for i in range(7501):
	transactions.append([str(data.values[i,j]) for j in range(0,20)])
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
res=list(rules)
print(res)
