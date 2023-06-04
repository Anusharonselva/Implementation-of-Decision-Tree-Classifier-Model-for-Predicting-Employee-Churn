# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

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
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.ANUSHARON
RegisterNumber:  212222240010

import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![decision tree classifier model](sam.png)

![Screenshot (214)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/1b63bb77-b875-449e-bcde-3de399ec7581)

![Screenshot (215)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/c7ae03c7-6a39-457c-b41f-302958acc202)

![Screenshot (216)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/767a32a2-fb67-4df9-8d40-67a8c1d9ebda)

![Screenshot (218)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/6882d730-c067-4892-9cfc-ac862f7657db)

![Screenshot (218)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/d26baf92-f32b-4af7-a293-c8ad5589aa53)

![218 1](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/553253f2-95dc-4dc2-93c4-42b7071988f1)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
