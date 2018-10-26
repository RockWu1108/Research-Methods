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
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import math
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns
'''(1)載入數據庫'''
data_url="insurance.csv"
df=pd.read_csv(data_url)
predictors=['age','sex','bmi','charges']

#資料預處理
label_encoder=preprocessing.LabelEncoder()
#性別(male=1,female=0)
df['sex']=label_encoder.fit_transform(df['sex'])
##抽菸(yes=1,no=0)
df['smoker']=label_encoder.fit_transform(df['smoker'])
##地區改為0,1(sw=3,se=2,nw=1,ne=0)
df['region']=label_encoder.fit_transform(df['region'])
X = pd.DataFrame(df,columns=['age','sex','charges','bmi'])
y = df["smoker"]


'''(2)描述性分析'''
print("描述性分析:\n",df[predictors].describe())

'''(3)散佈圖(倆倆變數)'''
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[predictors], size=2.5);
plt.tight_layout()
plt.show()


'''(4)相關矩陣'''
#資料正規化
df_scaled=pd.DataFrame(preprocessing.scale(df[predictors]),columns=predictors)
print("相關矩陣:\n",df_scaled[predictors].corr())


'''(5)共變異數矩陣'''
'''(6)eiqenvalue分解'''
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3,random_state=0)
# 標準化到單位變異數
sc=StandardScaler()
XTrain_std=sc.fit_transform(XTrain)
XTest_std=sc.fit_transform(XTest)
cov_mat=np.cov(XTest_std.T)

#eigen_vals=特徵值
#eigen_vecs=特徵向量
eig_vals,eig_vecs=np.linalg.eig(cov_mat)
print("\nEigenvalues: \n%s"% eig_vals)

#累加特徵值
tot=sum(eig_vals)
var_exp = [(i/tot) for i in sorted(eig_vals , reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('累加特徵值:\n',cum_var_exp)

#畫圖
plt.bar(range(1,5),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,5),cum_var_exp,where='mid',label='cumulative explained vaeiance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

#特征值對應的特征向量
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#eig_pairs.sort(reverse=True)
#feature=eig_pairs[0][1]
#print("="*50)
#print(feature)

'''(7)找出主成分矩陣'''

#w = np.hstack((eig_pairs[0][1][:, np.newaxis].real,
#              eig_pairs[1][1][:, np.newaxis].real,
#              eig_pairs[2][1][:, np.newaxis].real))

print('Matrix W:\n')
for i in range(3):
    w = np.hstack(eig_pairs[i][1][:, np.newaxis].real)
    print(w)
    
'''(8)不同數目主成分的MSE'''

for i in range(4):
   
    esor=PCA(n_components=i+1)
    pca_x_train=esor.fit_transform(XTrain_std)
    pca_x_test=esor.fit_transform(XTest_std)
    lm=LinearRegression()
    lm.fit(pca_x_train,yTrain)
    pca_y_predict=lm.predict(pca_x_test)
    mse=np.mean((pca_y_predict-yTest)**2)
    print("\nMSE:",i+1,":",mse)

'''(9)說明解釋量和eigenvalue和MSE的關係'''

print("eigenvalue和MSE的關係:"
    "\n當主成分從四維降至一維的解釋變異量約只有4成、MSE為0.1229、特徵值為1.483，"
    "\n當主成分從四維降至二維的解釋變異量約只有6成、MSE為0.1239、特徵值為0.616，"
    "\n當主成分從四維降至三維則可以達到8成的解釋變異量、MSE為0.12427、特徵值為1.055"
    "\n從上述中得知MSE皆約相等，雖四維降至一維的特徵值較高，但解釋性差，因此選擇主成分從四維降至三維")


'''用sklearn的PCA和回歸分析'''
print('Slope: %.3f' %lm.coef_[0])
print('Intercept: %.3f' % lm.intercept_)




#XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3,random_state=0)
## 標準化到單位變異數
#sc=StandardScaler()
#XTrain_std=sc.fit_transform(XTrain)
#XTest_std=sc.fit_transform(XTest)
#
#
#
##未使用套件
#cov_mat=np.cov(XTest_std.T)
##eigen_vals=特徵值
##eigen_vecs=特徵向量
#eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
#print("\nEigenvalues \n%s"% eigen_vals)
#print("\n未使用Sklearn套件:")
#tot=sum(eigen_vals)
#var_exp = [(i/tot) for i in sorted(eigen_vals , reverse=True)]
#cum_var_exp = np.cumsum(var_exp)
#
#plt.bar(range(1,5),var_exp,alpha=0.5,align='center',label='individual explained variance')
#plt.step(range(1,5),cum_var_exp,where='mid',label='cumulative explained vaeiance')
#plt.ylabel('Explained variance ratio')
#plt.xlabel('Principal components')
#plt.legend(loc='best')
#plt.show()
#print("\n使用Sklearn套件:")
##使用套件
#pca = PCA ()
#XTrain_pca = pca.fit_transform(XTrain_std)
#pca.explained_variance_ratio_
#plt.bar(range(1,5),pca.explained_variance_ratio_,alpha=0.5,align='center',label='individual explained variance')
#plt.step(range(1,5),np.cumsum(pca.explained_variance_ratio_),where='mid',label='cumulative explained vaeiance')
#plt.ylabel('Explained variance ratio')
#plt.xlabel('Principal components')
#plt.legend(loc='best')
#plt.show()
