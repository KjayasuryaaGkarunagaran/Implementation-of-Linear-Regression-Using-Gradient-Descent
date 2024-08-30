# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function. 
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Jayasuryaa k
RegisterNumber:  212222040060
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y, learing_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learing_rate*(1/len(x1)*x.T.dot(errors))
    return theta
data=pd.read_csv('/content/50_Startups.csv')
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
predictions=np.dot(np.append(1,new_scaled),theta)
predictions=predictions.reshape(-1,1)
pre=scaler.inverse_transform(predictions)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2024-08-28 112506](https://github.com/user-attachments/assets/2179e0b2-0cf7-40eb-9acf-0b0752bd0b49)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
