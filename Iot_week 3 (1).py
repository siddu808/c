#!/usr/bin/env python
# coding: utf-8

# Week 3:
# Basic plots for data exploration (Use Iris dataset)
# 1. Generate box plot for each of the four predictors.
# 2. Generate box plot for a specific feature
# 3. Generate histogram for a specific feature
# 4. Generate Scatter plot of petal length vs. sepal length

# In[133]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[134]:


data = pd.read_csv("C:/Users/saile/Desktop/IoT/datasets/iris.csv")


# In[135]:


print (data.head(10))


# In[136]:


data.describe()


# In[137]:


data.info()


# generate box plot for each of the four predictors

# In[168]:


import matplotlib.pyplot as plt
import seaborn as sns

#box plot for petal length

sns.boxplot(x='Species', y='Petal Length',data=data)


# In[169]:


#box plot for Sepal length

sns.boxplot(x='Species', y='Sepal Length',data=data)


# In[170]:


#box plot for Sepal Width

sns.boxplot(x='Species', y='Sepal Width',data=data)


# In[172]:


#box plot for petal Width

sns.boxplot(x='Species', y='Petal Width ',data=data)


# Observations
# 
# 1.Outliers are observed in petal length for sertosa and versicolor feature
# 2.in speal length and sepal width feature for virginica
# 3.Petal  width for sertosa
# 
# 

# In[151]:


#Histogram for Sepal Length
plt.figure(figsize = (10, 7))
x = data["Sepal Length"]

plt.hist(x, bins = 10, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")


 #2: Histogram for Sepal Width

plt.figure(figsize = (10, 7))
x = data['Sepal Width']
  
plt.hist(x, bins = 10, color = "blue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
  
plt.show()

#3: Histogram for petal length
plt.figure(figsize = (10, 7))
x = data['Petal Length']
plt.hist(x, bins = 10, color = "red")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length in cm")
plt.ylabel("Count")
plt.show()

#4: Histogram for petal Width
plt.figure(figsize = (10, 7))
x = data['Petal Width ']
plt.hist(x, bins = 20, color = "green")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width")
plt.ylabel("Count")
plt.show()


# In[156]:


#5 Data preparation for Box Plot
# removing Id column
new_data = data[["Sepal Length","Sepal Width","Petal Length","Petal Width "]]
print(new_data.head())
plt.figure(figsize = (10, 7))
new_data.boxplot()


# In[165]:


import pandas as pd
import matplotlib.pyplot as plt
fig = data[data.Species==0].plot(kind='scatter',x='Sepal Length',y='Sepal Width',color='orange', label='Setosa')
data[data.Species==1].plot(kind='scatter',x='Sepal Length',y='Sepal Width',color='blue', label='versicolor',ax=fig)
data[data.Species==2].plot(kind='scatter',x='Sepal Length',y='Sepal Width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Sepal Length")
fig.set_title("Petal Length VS Sepal Length")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# In[174]:


# box plot for a special feature

sns.boxplot(x='Species', y='Sepal Width',data=data)
plt.show()


# In[ ]:




