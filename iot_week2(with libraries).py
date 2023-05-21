#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','Chanchal','Gasper','Naviya','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,4,23,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

#Create a DataFrame
df = pd.DataFrame(d)
df


# In[2]:


print("Mean Values in the Distribution")
print(df.mean())


# In[3]:


print(df['Rating'].mean())


# In[4]:


print("Median Values in the Distribution")
print(df.median())


# In[5]:


print(df['Rating'].median())


# In[6]:


print("Median Values in the Distribution")
print(df.mode())


# In[7]:


df.std()


# In[10]:


df.var()


# In[13]:


#Interquartile range using numpy.median
# Import the numpy library as np
import numpy as np
  
data = [32, 36, 46, 47, 56, 69, 75, 79, 79, 88, 89, 91, 92, 93, 96, 97, 
        101, 105, 112, 116]
  
# First quartile (Q1)
Q1 = np.median(data[:10])
  
# Third quartile (Q3)
Q3 = np.median(data[10:])
  
# Interquartile range (IQR)
IQR = Q3 - Q1
  
print(IQR)


# In[15]:


#interquartile range using numpy.percentile

# Import numpy library
import numpy as np
  
data = [32, 36, 46, 47, 56, 69, 75, 79, 79, 88, 89, 91, 92, 93, 96, 97, 
        101, 105, 112, 116]
  
# First quartile (Q1)
Q1 = np.percentile(data, 25, interpolation = 'midpoint')
  
# Third quartile (Q3)
Q3 = np.percentile(data, 75, interpolation = 'midpoint')
  
# Interquaritle range (IQR)
IQR = Q3 - Q1
  
print(IQR)


# In[ ]:




