#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  classification_report
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
filterwarnings('ignore')


# In[3]:


data = pd.read_csv('C:/Users/saile/Desktop/IoT/datasets/model evaluation_spine dataset.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


# Data is clean except the "Unnamed: 13" column
data.info()


# In[7]:


# Type of Backbone Conditions
data.Class_att.unique()


# In[8]:


# Remove unwanted column
df = data.drop("Unnamed: 13", axis=1)


# In[9]:


# Change the Column names to be sensable
df.rename(columns = {
    "Col1" : "pelvic_incidence", 
    "Col2" : "pelvic_tilt",
    "Col3" : "lumbar_lordosis_angle",
    "Col4" : "sacral_slope", 
    "Col5" : "pelvic_radius",
    "Col6" : "degree_spondylolisthesis", 
    "Col7" : "pelvic_slope",
    "Col8" : "direct_tilt",
    "Col9" : "thoracic_slope", 
    "Col10" :"cervical_tilt", 
    "Col11" : "sacrum_angle",
    "Col12" : "scoliosis_slope", 
    "Class_att" : "target"}, inplace=True)


# In[10]:


# How skewed is the data?
df["target"].value_counts().sort_index().plot.bar()


# In[11]:


# Convert categorical to numeric {"Abnormal":0, Normal:1}
df.target = df.target.astype("category").cat.codes


# In[12]:


df.head()


# In[13]:


# 88% Accuracy
dataset = df[["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis","target"]]


# In[14]:


# Separate input and output
y = dataset.target
X = dataset.drop("target", axis=1)


# In[15]:


# Split data between train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


# List models
from sklearn import datasets
from sklearn.cluster import KMeans
models =  [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,GradientBoostingClassifier, SVC]


# In[17]:


# Train & Predict models
acc_list = []
name_list =[]
for model in models:
    clf = model()
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    name_list.append((model).__name__)
    acc_list.append(classification_report(y_test,predictions,output_dict=True)["accuracy"])
    print((model).__name__," --> ",classification_report(y_test,predictions,output_dict=True)["accuracy"])


# In[18]:


# Make a dataframe
team = pd.DataFrame(list(zip(name_list,acc_list)))  
team.columns =['Name', "Accuracy"]
team


# In[19]:


# Render a Chart
sns.barplot(x=team["Name"], y=team["Accuracy"],data=team)

# Rotate x-labels
plt.xticks(rotation=-45)
plt.ylim(0.7, 1)


# In[20]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[21]:


km = KMeans(n_clusters=2, random_state=42)
#
# Fit the KMeans model
#
km.fit_predict(X_train)
#
# Calculate Silhoutte Score
#
score = silhouette_score(X_train, km.labels_, metric='euclidean')
#
# Print the score
#
print('Silhouetter Score: %.3f' % score)


# In[22]:


pip install yellowbrick


# In[23]:


from yellowbrick.cluster import SilhouetteVisualizer
 
fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2,3,4,5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i,random_state=42)
    q, mod = divmod(i, 2)
    '''
   # Create SilhouetteVisualizer instance with KMeans instance
    #Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X_train)


# In[ ]:




