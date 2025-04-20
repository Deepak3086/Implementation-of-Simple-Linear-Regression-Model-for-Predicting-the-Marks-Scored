# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEEPAK JG 
RegisterNumber:  212224220019
*/
```

## Output:
```
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```
![image](https://github.com/user-attachments/assets/4adc28e5-ab86-4d83-9681-03e6e1272de8)

![image](https://github.com/user-attachments/assets/564a8e8d-109e-48f9-a7f1-792a734fadeb)

![image](https://github.com/user-attachments/assets/30e29430-a67b-4077-8a9b-5e1523ab727e)

![image](https://github.com/user-attachments/assets/1722fac5-508c-4d6a-88e3-003419f9d80e)

![image](https://github.com/user-attachments/assets/db1afb34-e9f2-4583-8e26-083f5a3e5ec5)

![image](https://github.com/user-attachments/assets/1ec83535-5ad3-4216-898c-d6d22bc5ba8b)

![image](https://github.com/user-attachments/assets/bee9c61e-faee-4ba4-9624-5ad59c5331c9)

![image](https://github.com/user-attachments/assets/3dcf8370-2148-4396-9f8f-9ab020225592)

![image](https://github.com/user-attachments/assets/068cd2ff-25b0-4d92-97db-db56485070de)

![image](https://github.com/user-attachments/assets/bf274976-721f-4738-81cc-02cb037f9b3b)

![image](https://github.com/user-attachments/assets/bc51ea5b-1020-411e-bea6-a592029189c6)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
