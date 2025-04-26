# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:

# Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: DIVYA M
# RegisterNumber:  212223040043
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset.info()
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary', axis=1)
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
dataset.info()
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
print(x_train.shape)
print(y_train.shape)
from sklearn.linear_model import LogisticRegression
cl=LogisticRegression(max_iter=1000)
cl.fit(x_train,y_train)
y_pred=cl.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
cl.predict([[0,87,0,95,0,2,8,0,0,1,5,6]])
cl.predict([[1,2,3,4,5,6,7,8,9,10,11,12]])
```

## Output:
![Screenshot 2025-04-26 215242](https://github.com/user-attachments/assets/92d1e80f-568c-4977-a1da-e2b077c2ad0c)
![Screenshot 2025-04-26 214844](https://github.com/user-attachments/assets/8d832669-1d11-4ab9-b6c9-9fcb62887930)
![Screenshot 2025-04-26 215300](https://github.com/user-attachments/assets/a6f5af77-dbab-4462-997e-cdf0008fdc5f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
