#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data =pd.read_excel('AS5.xls')


# In[ ]:


# Training SLR model using BGD algorithm.
#Dataset: Pressure and Weight in Cryogenic Flow Meters


# In[3]:


data.head(34)


# In[ ]:


#NORMALIZATION


# In[4]:


normalized_data = (data-data.mean())/data.std()
normalized_data.head()


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(normalized_data.X, normalized_data.Y, test_size = 0.10, random_state = 42)


# In[7]:


m = 1 
c = -1 
lr = 0.01 
delta_m = 1 
delta_c = 1 
max_iters = 1000 
iters_count = 0

def deriv(m_f, c_f, x, y):
  m_deriv = -1*(y-m_f*x-c_f)*x
  c_deriv = -1*(y-m_f*x-c_f)
  return m_deriv, c_deriv  


while iters_count < max_iters:
  for i in range(x_train.shape[0]):
    delta_m, delta_c = deriv(m, c, x_train.iloc[i], y_train.iloc[i])
    delta_m = -lr * delta_m
    delta_c = -lr * delta_c
    m += delta_m
    c += delta_c
  iters_count += 1
  print(f"Iteration: {iters_count}\tValue of m: {m}, \tValue of c: {c}")

print(f"\nThe local minima occurs at: {m}, {c}")


# In[8]:


import numpy as np

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[9]:


y_pred_train = []
for i in x_train:
  y_p_tr = (m * i) + c
  y_pred_train.append(y_p_tr)
y_pred_train = np.array(y_pred_train)


# In[10]:


y_pred_test = []
for i in x_test:
  y_p_te = (m * i) + c
  y_pred_test.append(y_p_te)
y_pred_test = np.array(y_pred_test)


# In[11]:


import math
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error


#Training Accuracies
mse = math.sqrt(mean_squared_error(y_train, y_pred_train)) 
print('Root mean square error', mse) 
mse = (mean_squared_error(y_train, y_pred_train)) 
print('Mean square error', mse) 
mae=mean_absolute_error(y_train, y_pred_train)
print('Mean absolute error', mae)


# In[12]:


#Testing Accuracies
mse = math.sqrt(mean_squared_error(y_test, y_pred_test)) 
print('Root mean square error', mse) 
mse = (mean_squared_error(y_test, y_pred_test)) 
print('Mean square error', mse) 
mae=mean_absolute_error(y_test, y_pred_test)
print('Mean absolute error', mae)


# In[ ]:




