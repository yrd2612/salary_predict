#!/usr/bin/env python
# coding: utf-8

# Salary vs years of experiance

# In[1]:

import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
# import joblib


# In[2]:


import pandas as pd
from sympy import Symbol, Derivative


# In[3]:
st.title("Predict your salary")

x = st.number_input('Enter years of experiance')
data = pd.read_csv('Salary_Data.csv')


# In[4]:


X = data["YearsExperience"]
y = data["Salary"]


# In[5]:


x_train = X.to_numpy()
y_train = y.to_numpy()


# In[6]:


print(x_train)
print(y_train)


# In[7]:


print(x_train.shape)
print(y_train.shape)


# In[8]:


def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        fwb_i = (np.dot(w,x[i]) + b)
        cost = cost + (fwb_i - y[i])**2
    cost = cost/(2*m)
    return cost


# In[9]:


def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        err = ((np.dot(X[i], w) + b) - y[i])
        # finding a new type of error
        
        #err = ((np.dot(X[i], w) + b) - y[i])/m
        dj_dw_i = err*x[i]
        dj_dw += dj_dw_i
        dj_db = err + dj_db
        #print('dj_db is : ',dj_db,' dj_dw is : ',dj_dw, ' error is : ',err)
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db,dj_dw


# In[10]:


def gradient_descent(x,y,w,b):
    j_history =[]
    alpha = 0.005
    for i in range(1000):
        dj_db,dj_dw = compute_gradient(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        j_history.append(compute_cost(x,y,w,b))
    return w,b,j_history


# In[ ]:


w_in = 500
b_in = 1
w_final,b_final,j_history = gradient_descent(x_train,y_train,w_in,b_in)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")


# final result

# In[12]:


m = x_train.shape[0]
pred = []
pred_sum = 0
actual_sum = 0
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    pred.append(np.dot(x_train[i], w_final) + b_final)
    pred_sum += pred[i]
    actual_sum += y_train[i]
accuracy = (pred_sum/actual_sum) * 100
print(accuracy)
print('Standard deviation is : ')
mean = pred_sum/m

std = 0
for i in range(m):
    std_1 = (pred[i] - mean) ** 2
    print('std_1',std_1)
    std_2 = (std/m) ** 0.5
    std = std + std_2
print(std)

    


# In[13]:


def actual():
    m = x.shape[0]
    for i in range(m):
        y = y_train[i]
        y_act.append(y)


# In[14]:


def predict(x):
    # print('prediction is :',w_final*x + b_final)
    return w_final*x + b_final 
    
    
    


# In[ ]:


print('Enter ')
#x = float(input())
prediction = 0
prediction = predict(x)
#print(prediction)


# In[ ]:


st.button("Submit")
st.text('Expected salary is :')
st.write(predict(x))
