# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:11:59 2018
@author: Rock
"""
"""
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, 
     weights that are relatively high or low relative to height, 
     objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import ensemble, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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


XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3,random_state=0)
#标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。

sc=StandardScaler()
XTrain=sc.fit_transform(XTrain)
XTest=sc.fit_transform(XTest)

lda = LDA(n_components=3)
XTrain=lda.fit_transform(XTrain,yTrain)
XTest= lda.transform(XTest)
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(XTrain, yTrain)  
y_pred = classifier.predict(XTest) 
print("降維度後RandomForest:")
print('Accuracy:' + str(accuracy_score(yTest, y_pred))+'\n') 


xTrain, xTest, ytrain, ytest = train_test_split(X, y, test_size=0.3,random_state=0)
'''隨機森林'''
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(xTrain, ytrain)  
y_pred = classifier.predict(xTest) 
print("降維度前RandomForest:")
print('Accuracy:' + str(accuracy_score(yTest, y_pred))+'\n') 





