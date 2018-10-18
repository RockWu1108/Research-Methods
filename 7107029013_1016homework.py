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
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing,linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import math
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import neighbors
import matplotlib.pyplot as plt

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
# 標準化到單位變異數
sc=StandardScaler()
XTrain_std=sc.fit_transform(XTrain)
XTest_std=sc.fit_transform(XTest)



#未使用套件
cov_mat=np.cov(XTest_std.T)
#eigen_vals=特徵值
#eigen_vecs=特徵向量
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print("\nEigenvalues \n%s"% eigen_vals)
print("\n未使用Sklearn套件:")
tot=sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals , reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,5),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,5),cum_var_exp,where='mid',label='cumulative explained vaeiance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
print("\n使用Sklearn套件:")
#使用套件
pca = PCA ()
XTrain_pca = pca.fit_transform(XTrain_std)
pca.explained_variance_ratio_
plt.bar(range(1,5),pca.explained_variance_ratio_,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,5),np.cumsum(pca.explained_variance_ratio_),where='mid',label='cumulative explained vaeiance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()



