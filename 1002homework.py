# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:47:53 2018

@author: Rock
"""
import pandas as pd
from scipy import stats
import numpy as np
data_url="insurance.csv"
df=pd.read_csv(data_url)
import math

#'''母體平均數'''
#for x in range(1000):
#    Age_sample=np.random.choice(df['age'],size=100)
#    Bmi_sample=np.random.choice(df['bmi'],size=100)
#    Charges_sample=np.random.choice(df['charges'],size=100)
#    age.append(Age_sample.mean())
#    bmi.append(Bmi_sample.mean())
#    charges.append(Charges_sample.mean())
#print("Age母體平均數:",sum(age)/1000.0)
#print("Bmi母體平均數:",sum(bmi)/1000.0)
#print("Charges母體平均數:",sum(charges)/1000.0)


#'''母體平均數'''
print("Age母體平均數:",sum(df['age'])/df['age'].count())
print("Bmi母體平均數:",sum(df['bmi'])/df['bmi'].count())
print("Charges母體平均數:",sum(df['charges'])/df['charges'].count(),'\n')

#'''樣本平均數'''
#隨機抽樣大樣本
Age_sample=np.random.choice(a=df['age'],size=500)
Bmi_sample=np.random.choice(a=df['bmi'],size=500)
Charges_sample=np.random.choice(a=df['charges'],size=500)

#計算樣本平均
Age_sample_mean = Age_sample.mean()
Bmi_sample_mean = Bmi_sample.mean()
Charges_sample_mean = Charges_sample.mean()
print("隨機抽樣500筆資料:")
print('Age樣本平均數:',Age_sample_mean)
print('Bmi樣本平均數:',Bmi_sample_mean)
print('Charges樣本平均數:',Charges_sample_mean,'\n')

#樣本標準差
Age_sample_std = Age_sample.std()
Bmi_sample_std = Bmi_sample.std()
Charges_sample_std = Charges_sample.std()
print('Age樣本標準差:',Age_sample_std)
print('Bmi樣本標準差:',Bmi_sample_std)
print('Charges樣本標準差:',Charges_sample_std,'\n')

#樣本推估母體標準差
Age_sigma=Age_sample_std/math.sqrt(500-1)#自由度-1
Bmi_sigma=Bmi_sample_std/math.sqrt(500-1)
Charges_sigma=Charges_sample_std/math.sqrt(500-1)
print('Age樣本推估母體標準差:',Age_sigma)
print("Bmi樣本推估母體標準差:",Bmi_sigma)
print("Charges樣本推估母體標準差:",Charges_sigma,'\n')

#樣本數大於30採用Z分數(1-α)
'''95%信心水準'''
z_critical  = stats.norm.ppf(q=0.975)
print("95%信心水準下:")
print("Z分數:",z_critical)

#信賴區間估計 stats.norm.interval(alpha=信心水準,loc=平均數,scale=標準差)
age_conf_int=stats.norm.interval(alpha=0.95,loc=Age_sample_mean,scale=Age_sigma)
bmi_conf_int=stats.norm.interval(alpha=0.95,loc=Bmi_sample_mean,scale=Bmi_sigma)
charges_conf_int=stats.norm.interval(alpha=0.95,loc=Charges_sample_mean,scale=Charges_sigma)
print("Age信賴區間範圍:",age_conf_int[0],"~",age_conf_int[1])
print("Bmi信賴區間範圍:",bmi_conf_int[0],"~",bmi_conf_int[1])
print("Charges信賴區間範圍:",charges_conf_int[0],"~",charges_conf_int[1])
print("="*70,'\n')
'''=========================================================================='''






#小樣本(樣本數低於30)，採用T分配

Age_small_sample=np.random.choice(a=df['age'],size=25)
Bmi_small_sample=np.random.choice(a=df['bmi'],size=25)
Charges_small_sample=np.random.choice(a=df['charges'],size=25)
#計算樣本平均
Age_small_sample_mean = Age_small_sample.mean()
Bmi_small_sample_mean = Bmi_small_sample.mean()
Charges_small_sample_mean = Charges_small_sample.mean()
print("隨機抽樣25筆資料:")
print('Age樣本平均數:',Age_small_sample_mean)
print('Bmi樣本平均數:',Bmi_small_sample_mean)
print('Charges樣本平均數:',Charges_small_sample_mean,'\n')

#樣本標準差
Age_small_sample_std = Age_small_sample.std()
Bmi_small_sample_std = Bmi_small_sample.std()
Charges_small_sample_std = Charges_small_sample.std()
print('Age樣本標準差:',Age_small_sample_std)
print('Bmi樣本標準差:',Bmi_small_sample_std)
print('Charges樣本標準差:',Charges_small_sample_std,'\n')

#樣本推估母體標準差
Age_small_sigma=Age_small_sample_std/math.sqrt(25-1)#自由度-1
Bmi_small_sigma=Bmi_small_sample_std/math.sqrt(25-1)
Charges_small_sigma=Charges_small_sample_std/math.sqrt(25-1)
print('Age樣本推估母體標準差:',Age_small_sigma)
print("Bmi樣本推估母體標準差:",Bmi_small_sigma)
print("Charges樣本推估母體標準差:",Charges_small_sigma,'\n')


#樣本數小於30採用T分數
'''95%信心水準'''
T_critical  = stats.t.ppf(q=0.975,df=25-1)#df=自由度
print("95%信心水準下:")
print("T分數:",T_critical)

#信賴區間估計 stats.norm.interval(alpha=信心水準,df=自由度,loc=平均數,scale=標準差)
age_small_conf_int=stats.t.interval(alpha=0.95,df=25-1,loc=Age_small_sample_mean,scale=Age_small_sigma)
bmi_small_conf_int=stats.t.interval(alpha=0.95,df=25-1,loc=Bmi_small_sample_mean,scale=Bmi_small_sigma)
charges_small_conf_int=stats.t.interval(alpha=0.95,df=25-1,loc=Charges_small_sample_mean,scale=Charges_small_sigma)
print("Age信賴區間範圍:",age_small_conf_int[0],"~",age_conf_int[1])
print("Bmi信賴區間範圍:",bmi_small_conf_int[0],"~",bmi_conf_int[1])
print("Charges信賴區間範圍:",charges_small_conf_int[0],"~",charges_conf_int[1])



#T檢定
#假設母體平均數
#Age_population_mean=40
Bmi_population_mean=31
#Charges_population_mean=13500

#t_age_obtained = (Age_small_sample_mean-Age_population_mean)/Age_small_sigma
t_bmi_obtained = (Bmi_small_sample_mean-Bmi_population_mean)/Bmi_small_sigma
#t_charges_obtained = (Charges_small_sample_mean-Charges_population_mean)/Charges_small_sigma


#建立假設
#H0=年齡約平均45歲
#H1=年齡平均不等於45歲
print("\n假設Bmi:\nH0=Bmi平均31\nH1=Bmi平均不等於31")
t_bmi_critical=stats.t.ppf(q=0.975,df=30-1)
print("Bmi的T檢定統計量: ",t_bmi_obtained)
print('Bmi T分數:',t_bmi_critical)
if(math.sqrt(math.pow(t_bmi_obtained,2))>=t_bmi_critical or math.sqrt(math.pow(t_bmi_obtained,2))<=-(t_bmi_critical)):
    print("拒絕虛無假設")
else:
    print('接受虛無假設')


