# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:35:45 2018

@author: Rock
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

data_url="insurance.csv"
df=pd.read_csv(data_url)
predictors=['age','sex','bmi','children']
'''(2)更改類別值'''
label_encoder=preprocessing.LabelEncoder()
#性別(male=1,female=0)
df['sex']=label_encoder.fit_transform(df['sex'])
##抽菸(yes=1,no=0)
df['smoker']=label_encoder.fit_transform(df['smoker'])
##地區改為0,1(sw=3,se=2,nw=1,ne=0)
df['region']=label_encoder.fit_transform(df['region'])

X = pd.DataFrame(df,columns=['age','sex','charges','bmi'])
y = df["smoker"]

#切割訓練測試集
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3,random_state=0)

model = GaussianNB()
model.fit(X, y)
expected = y
# make predictions
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


