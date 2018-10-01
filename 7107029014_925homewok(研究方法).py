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
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_url="insurance.csv"
df=pd.read_csv(data_url)
print(df)
'''(1)描術性統計'''
print("描述性統計:\n",df.describe())

predictors=['age','sex','bmi','children']


'''(2)更改類別值'''
#df.loc[df['sex']=='male','sex']=0
#df.loc[df['sex']=='female','sex']=1
##地區改為0,1(sw=0,se=1,nw=2,ne=3)
#df.loc[df['region']=='southwest','region']=0
#df.loc[df['region']=='southeast','region']=1
#df.loc[df['region']=='northwest','region']=2
#df.loc[df['region']=='northeast','region']=3
##是否抽菸(yes=1,no=0)
#df.loc[df['smoker']=='yes','smoker']=1
#df.loc[df['smoker']=='no','smoker']=0

label_encoder=preprocessing.LabelEncoder()
#性別(male=1,female=0)
df['sex']=label_encoder.fit_transform(df['sex'])
##抽菸(yes=1,no=0)
df['smoker']=label_encoder.fit_transform(df['smoker'])
##地區改為0,1(sw=3,se=2,nw=1,ne=0)
df['region']=label_encoder.fit_transform(df['region'])
print(df)

"""(3)相關係數"""
print("相關係數:\n",df[predictors].corr())

'''(4)資料正規化(資料介於-1~1)'''
df_scaled=pd.DataFrame(preprocessing.scale(df[predictors]),columns=predictors)
print("資料正規化:\n",df_scaled.head())
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df_scaled[predictors], size=2.5);
plt.tight_layout()
plt.show()


'''(5)資料正規化MaxMin(資料介於0~1)'''
scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
np_minmax=scalar.fit_transform(df[predictors])
df_minmax=pd.DataFrame(np_minmax,columns=predictors)
print("資料正規化(Max,Min):\n",df_minmax.head())
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df_minmax[predictors], size=2.5);
plt.tight_layout()
plt.show()


'''線性回歸'''
X = df['bmi'].values.reshape(-1,1)
#target = pd.DataFrame(df, columns=['charges'])
y = df['charges'].values.reshape(-1,1)
#lm = LinearRegression()
#lm.fit(X, y)
#print("迴歸係數:", lm.coef_)
#print("截距:", lm.intercept_ )
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return 

lin_regplot(X, y, slr)
plt.xlabel('Bmi')
plt.ylabel('charges')
plt.tight_layout()
# plt.savefig('./figures/scikit_lr_fit.png', dpi=300)
plt.show()
'''MSE'''
mse=np.mean((y_pred-y)**2)
print('MSE:',mse)


