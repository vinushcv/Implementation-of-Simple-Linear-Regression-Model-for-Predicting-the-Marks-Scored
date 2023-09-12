# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning dataset values.
3. Import LinearRegression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given datas.

 
## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: vinush.cv  
RegisterNumber:  212222230176
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("student_scores.csv")
dataset.head()

X = dataset.iloc[:,:-1].values
X
y = dataset.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

## Output:
df.head():
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/4930f29b-bb2f-4f86-b0bc-ad1a25dcf176)

df.tail():
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/32697766-bf40-4b80-be53-b29e57747769)

Array value of X:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/ee56ae92-3779-462a-8fd3-94f925b332de)

Array value of Y:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/c5ea284d-dd7f-4583-bda5-d00371059fb4)

Values of Y prediction and Values of Y prediction:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/2cdf4f04-05cb-444e-894f-a3a60a953d57)

Training set graph:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/dfe92074-3592-4798-83a2-f65c330ff18c)

Test set graph:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/c62d4eb2-1092-479d-b9ef-eec8c2ead530)

Values of MSE, MAE and RMSE:
![image](https://github.com/vinushcv/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113975318/f156b744-86e5-4c1d-9ee6-81f9ef7f09ed)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
