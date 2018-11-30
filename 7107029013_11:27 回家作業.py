# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:52:31 2018

@author: user
"""
#x=[sunny,cloudy,rainy]
#i=0~2
X=[0.63,0.17,0.20]
# i=0~2 , j=0~2
SCR=[[0.5,0.375,0.125],
     [0.25,0.125,0.625],
     [0.25,0.375,0.375]]
# i=0~2 , j=0~3
Humidity=[[0.6,0.2,0.15,0.05],
          [0.25,0.25,0.25,0.25],
          [0.05,0.10,0.35,0.50]]
S=["Sunny","Cloudy","Rainy"]
temp1=[]
temp2=[]
temp3=[]

def T1():
    for i in range(3):
        temp1.append(round(X[i]*Humidity[i][0],7))
        print("a1(",S[i],")=",temp1[i])
    print("argMax(a1,j)= a1(",S[temp1.index(max(temp1))],")=",max(temp1),"\n")

def T2():
    for i in range(3): 
        count=0 
        for j in range(3):
            count=count+round(temp1[j]*SCR[j][i],7)*Humidity[i][1]
        temp2.append(count)                      
        print("a2(",S[i],")=",temp2[i])
    print("argMax(a2,j)= a2(",S[temp2.index(max(temp2))],")=",max(temp2),"\n")

def T3():   
    for i in range(3): 
        count=0 
        for j in range(3):
            count=count+round(temp2[j]*SCR[j][i],7)*Humidity[i][2]
        temp3.append(count)                      
        print("a3(",S[i],")=",temp3[i])
    print("argMax(a3,j)= a3(",S[temp3.index(max(temp3))],")=",max(temp3),"\n")

if __name__ == '__main__':
    T1()
    T2()
    T3()

