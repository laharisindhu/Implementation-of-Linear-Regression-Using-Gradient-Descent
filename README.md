# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations of gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: G.Lahari sindhu
RegisterNumber: 212223240038

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01, num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
theta=linear_regression(x1_Scaled,y1_Scaled)
new_data=np.array([165349.2,136897.8,471781.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/749bb850-9114-4f11-b054-c4f71b9926bb)
## X Values:
![image](https://github.com/user-attachments/assets/c698e435-81e0-46af-8b40-8eefed3d9d17)
## Y Values:
![image](https://github.com/user-attachments/assets/1e31f785-5405-4588-a2b7-1679165c886b)
## X Scaled values:
![image](https://github.com/user-attachments/assets/a79243d2-c482-4b67-98b4-1f32e5b28ff7)
## Y Scaled values:
![image](https://github.com/user-attachments/assets/03c2bbbf-575e-4dfd-811e-d55cf80cd603)
## Predicted Value:
![image](https://github.com/user-attachments/assets/877d8775-2590-4d4c-9b17-846c1069c76d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
