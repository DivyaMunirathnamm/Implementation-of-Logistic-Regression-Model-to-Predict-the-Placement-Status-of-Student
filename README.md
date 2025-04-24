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
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![Screenshot (854)](https://github.com/user-attachments/assets/d97000d7-532b-451a-bc26-c7b94c1dfafd)

![Screenshot (855)](https://github.com/user-attachments/assets/424d79a4-c0d6-47df-95db-45dbab36fd39)

![Screenshot (856)](https://github.com/user-attachments/assets/fbc0bc13-b014-4a12-a5b3-628531893f32)

![Screenshot (857)](https://github.com/user-attachments/assets/b8c49d5e-f263-4459-8170-c04dc359508c)

![Screenshot (858)](https://github.com/user-attachments/assets/d20edb72-9807-4676-8315-e56b9d7e84a9)

![Screenshot (859)](https://github.com/user-attachments/assets/9ff53bde-89e3-47bc-861a-5f2053095278)

![Screenshot (860)](https://github.com/user-attachments/assets/6d01c989-e101-4a25-8181-f0fe3b0711b9)

![Screenshot (861)](https://github.com/user-attachments/assets/5a709ba7-50d2-4a8a-944a-d41b3a93deb7)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
