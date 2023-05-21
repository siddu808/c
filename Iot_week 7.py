#!/usr/bin/env python
# coding: utf-8

# # week 7
# 
# 
# Feature Construction: (Use packages that are applicable)
# 1. Dummy coding categorical(nominal) variables.
# 2. Encoding categorical(ordinal) variables.
# 3. Transforming numeric(continuous)features to categorical features
# Feature Extraction: (Use packages that are applicable)
# 1. Principal Component Analysis (PCA)
# 2. Singular Value Decomposition (SVD)
# 3. Linear Discriminant Analysis (LDA)
# 4. Feature Subset Selection
# Compiler Design
# 
# Data set :IRIS 
# attributes: petal length, petal width, sepal length, sepal width
# 

# In[12]:


#reading the files
import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/saile/Desktop/IoT/datasets/iris.csv")


# In[13]:


df


# # dummy coding categorical variables

# In[15]:


#using pandas to create dummy variables
dummies =pd.get_dummies(df.Species)


# In[16]:


dummies


# In[18]:


#merging dummy and original dataset
Iris =pd.concat([df,dummies],axis='columns')
Iris


# In[19]:


Iris =Iris.drop(['Species'],axis='columns')
Iris


# In[20]:


#labelling encoding of nomial data usig skelarn 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
iris_1=df
iris_1.Species=le.fit_transform(iris_1.Species)
iris_1


# # 2 encoding categorical variables
# # 3 transforming numerical features to categorical features

# In[34]:


import matplotlib.pyplot as plt
x=df['Petal Width ']
plt.hist(x)
plt.show


# In[42]:


df['Petal category']=pd.cut(df['Petal Width '] ,bins=[0,1,2,2.5], labels=['small','medium','long'])
print(df['Petal category'])
print(df['Petal category'].value_counts())


# # Observation
# since categorization of the flower can be done through the petal width we have categorised the continous data into 3 categories as small medium and long

# # Encoding categorical variables

# In[47]:


from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
df['Petal category'] =encoder.fit_transform(df[['Petal category']])
df


# # observations
# 
# scikitlearn Ordinal Encoder is used to convert the categorical data to continous data

# # feature extraction
# 
# Feature Extraction: (Use packages that are applicable)
# 1. Principal Component Analysis (PCA)
# 2. Singular Value Decomposition (SVD)
# 3. Linear Discriminant Analysis (LDA)
# 4. Feature Subset Selection

# In[49]:


# read the csv files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
df= datasets.load_iris()


# In[50]:


df


# # Principle component analysis
# 

# In[52]:


# Determine the initial dimension of the data
X=df.data
Y=df.target
print(X.shape,Y.shape)


# In[54]:


#PCA for taregt dimension of the dataset
pca=PCA(n_components =2)
pca.fit(X)


# In[55]:


#visualizing  principle components
pca.components_


# In[56]:


#transforming the data from 4-D to 2-D using PCA
z=pca.transform(X)
z.shape


# In[59]:


#scatter plot
plt.scatter(z[:,0],z[:,1],c=Y)


# In[61]:


#variance ratio of the target dimensions

pca.explained_variance_ratio_


# # observations 
# 
# 4-D data converted to 2-D data
# Total variance of the data is equal to the sum of variance of the principle components

# # 2 .Linear Discriminant analysis(LDA)

# In[62]:


# Determine the initial dimension of the data
X=df.data
Y=df.target
print(X.shape,Y.shape)


# In[64]:


#Decomposing 4D to 2D using LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
X_r2=lda.fit(X,Y).transform(X)


# In[65]:


#getting the variance ratio
lda.explained_variance_ratio_


# In[69]:


#visualizing the 2d data in the form of scater plot
colors=['red','green','blue','yellow']
vectorizer=np.vectorize(lambda x:colors[x % len(colors)])
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(Y))


# # Observation
# 
# 4-D data converted to 2-D data using LDA

# # Feature subset selection
#  Filter Approach

# In[75]:


iris= pd.read_csv("C:/Users/saile/Desktop/IoT/datasets/iris.csv")

#visualizing using pair plot
import seaborn as sns
sns.pairplot(iris.drop(['Id'],axis =1),
             hue='Species',height=2)


# In[76]:


#visualization correlation using heatmap
sns.heatmap(iris.corr(method='pearson').drop(['Id'],axis =1).drop(['Id'],axis =0),annot=True);
plt.show()


# # Observations
# In filter approach we use statistics in feature selction
# Therefore pearsons correlation is used in the above to selct features with high correlation
# Since petal leghth and petal width have  high correlation ,these two can be used to classify the flower species
# 
# 

# In[ ]:




