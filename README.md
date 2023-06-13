# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.


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

1.data.head()

![Screenshot (295)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/5db83970-b8f8-49a2-9a42-bf6d46d7e727)


2.data.info()

![Screenshot (296)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/918679d9-ceeb-4c98-94d0-ee24de00d941)

3.isnull() and sum()

![Screenshot (297)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/0f014662-5fcf-4e3b-947f-87ed265f15b2)

4.data value counts()

![Screenshot (298)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/8d23892f-dbeb-4cb9-9767-4904a03026b1)

5.data.head() for salary

![Screenshot (299)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/a29b2321-6eea-40dc-b89f-c8265b706e06)

6.x.head()

![Screenshot (299) 1](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/08a6b738-e4ae-4c7a-b5a4-2849273910dd)

7.accuracy value

![Screenshot (299) 2](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/6c902442-ec5b-4d64-b247-2ca83710ce7e)

8.data prediction

![Screenshot (300)](https://github.com/Anusharonselva/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119405600/daf4ce2a-0193-49fb-ad61-7808f12f1922)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
