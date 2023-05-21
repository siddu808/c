#!/usr/bin/env python
# coding: utf-8

# #  WEEK 4 
# 
# 
# 
# Title: Auto-Mpg Data
# 
# Sources:
# (a) Origin: This dataset was taken from the StatLib library which is
# maintained at Carnegie Mellon University. The dataset was
# used in the 1983 American Statistical Association Exposition.
# (c) Date: July 7, 1993
# 
# Attribute Information:
# 
# mpg: continuous
# cylinders: multi-valued discrete
# displacement: continuous
# horsepower: continuous
# weight: continuous
# acceleration: continuous
# model year: multi-valued discrete
# origin: multi-valued discrete
# car name: string (unique for each instance)
# 
# 

# # Removing outliers/Missing values

# In[1]:


import pandas as pd
df = pd.read_csv('C:/Users/saile/Desktop/IoT/datasets/auto-mpg.csv')


# In[2]:


df


# In[4]:


#replacing  question marks with  null values in horse power
for col in df.columns:
    df[col].replace('?',None, inplace=True)
df


# In[6]:


#identifying missing values
df.isnull().sum()


# # Detect and remove outliers using percentiles- method_1

# In[10]:


#identiy the outliers in dispalcement
max_threshold=df['cylinders'].quantile(0.95)
min_threshold=df['cylinders'].quantile(0.05)
df[(df['cylinders']>max_threshold) |(df['cylinders']<min_threshold)]


# In[14]:


#dataframe after removing outliers

df[(df['cylinders']>max_threshold) |(df['cylinders']<min_threshold)]
df


# # outlier detetcion using standard deviation method

# In[18]:


#identifying outliers in mpg
df.acceleration.mean()
df.acceleration.std()
max1=df.acceleration.mean() + 3*df.acceleration.std()
min1=df.acceleration.mean() - 3*df.acceleration.std()
print("the max value is", max1)
print("the min value is", min1)


# In[20]:


#removing outliers
df[(df['acceleration']>max1) |(df['acceleration']<min1)]


# # Outlier detection using  z=method
# 

# In[27]:


df['zscore'] = df.acceleration- df.acceleration.mean()/df.acceleration.std()
df.head(5)


# In[31]:


df[(df.zscore<-3) |(df.zscore >3)]


# In[32]:


df[df['zscore'] >3]


# # outlier detection using IQR

# In[34]:


#identifying outliers
Q1= df.mpg.quantile(0.25)
Q3= df.mpg.quantile(0.75)
IQR= Q3-Q1
lower= Q1 -1.5*IQR
upper=Q3 +1.5*IQR
df[(df.mpg<lower)|(df.mpg>lower)]


# In[35]:


#removing outliers
df_no_outlier = df[(df.mpg<lower)|(df.mpg>lower)]
df_no_outlier


# In[36]:


newdf=df.dropna()
newdf


# # observations
# Missing values are found in horsepower columns and they are remived from the  dataframe using dropna() method
# Outliers are identified using various methods and in columns mpg,cylinder, and acceleration
# 

# # Inputing standard values

# In[ ]:


std =df[(df['cylinders'] <= max_threshold) | (df['cylinders']>=min_threshold)]
print('standard value',std)


# In[41]:


df[(df['cylinders']>max_threshold) |(df['cylinders']<min_threshold)]


# # observations
#  Outliers are imputed with standard mean value which were identified  using threshold values.
#  

# # capping of values

# In[43]:


df['acceleration'].clip(min1,max1, inplace=True)
df[(df.acceleration >max1) |(df.acceleration<min1)]


# #observations
# 
# 1.outliers in acceleration are capped with maximum and minimum  values respectively using clip method in pandas.
# 2.Clip methods replaces the value  with lower limit which is less than it and similarly works for values greater tahn upper limit

# In[ ]:




