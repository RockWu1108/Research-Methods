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
from scipy import stats
from sklearn import preprocessing
from sklearn.tree import export_graphviz
import math
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
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



'''決策樹'''
#ID3 Algo
XTrain_ID3, XTest_ID3, yTrain_ID3, yTest_ID3 = train_test_split(X, y, test_size=0.25,random_state=0)
ID3_tree = DecisionTreeClassifier(criterion='entropy')
ID3_tree.fit(XTrain_ID3, yTrain_ID3)
ID3_predict=ID3_tree.predict(XTest_ID3)

#CART ALGO
XTrain_Gini, XTest_Gini, yTrain_Gini, yTest_Gini = train_test_split(X, y, test_size=0.25,random_state=0)
CART_tree = DecisionTreeClassifier(criterion='gini')
CART_tree.fit(XTrain_Gini, yTrain_Gini)
CART_predict=CART_tree.predict(XTest_Gini)

#計算交叉表
#Yes_Yes=0
#No_No=0
#Yes_No=0
#No_Yes=0
#for ID3,CART in zip(ID3_predict,CART_predict):
#    if(ID3==1 and CART==1):
#        Yes_Yes+=1
#    if(ID3==0 and CART==0):
#        No_No+=1
#    if(ID3==1 and CART==0):
#        Yes_No+=1
#    if(ID3==0 and CART==1):
#        No_Yes+=1
        
voters = pd.DataFrame({"ID3":ID3_predict,
                       "CART":CART_predict})
#print(voters)
voter_tab = pd.crosstab(voters.ID3, voters.CART,margins=True)
#voter_tab.columns = ["YES", "NO", "小計"]
#voter_tab.index = ["YES", "NO", "小計"]
observed = voter_tab.iloc[0:3, 0:3]
print(observed)

#麥內碼檢定
#(B-C)/(√B+C)
#假設α=0.05

print("\n")
B=voter_tab.loc[0][1]
C=voter_tab.loc[1][0]
Z_value=(B-C)/(math.sqrt(B+C))
Z_Critical=stats.norm.ppf(0.975)
print("假設:")
print("H0:ID3和CART的效能一樣")
print("H1:ID3和CART的效能不同")
print("麥內碼Z檢定統計量:",Z_value)
print("α0.05 = > z0.05:",Z_Critical)
if(Z_value>Z_Critical or Z_value<-(Z_Critical)):
    print("結論顯著，ID3和CART效能不同")
else:
    print("結論不顯著，ID3和CART效能一樣")




