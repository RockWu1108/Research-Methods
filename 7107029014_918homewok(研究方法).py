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
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
data_url="insurance.csv"
df=pd.read_csv(data_url)
print(df.head(6))#印出資料前n筆
print(df.keys())#欄位名稱
X=pd.DataFrame(df,columns=['age', 'sex','bmi' , 'children','charges'])
print(X.head(6))

"""畫圖"""
"""Context=paper，notebook, talk, poster 切割依序由大到小"""
sns.set(style='whitegrid', context='paper')
cols = ['age', 'sex','bmi' , 'children','charges']
sns.pairplot(X[cols], size=2.5) #Pairplot(欄位,顯示大小)
plt.tight_layout()
plt.show()




