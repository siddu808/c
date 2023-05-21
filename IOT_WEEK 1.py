#!/usr/bin/env python
# coding: utf-8

# In[1]:


#manually entering the data
#import the pandas library as create an alias name as pd
import pandas as pd


# In[2]:


#Constructing DataFrame from a dictionary.
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df
  


# In[3]:


#Constructing DataFrame from a dictionary including Series:
d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
pd.DataFrame(data=d, index=[0, 1, 2, 3])
  


# In[6]:


#Constructing DataFrame from numpy ndarray:
import numpy as np
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),columns=['a', 'b', 'c']) 
df2
   


# In[9]:


#read the dataset
#option 1
data=pd.read_csv('C:/Users/saile/Desktop/ml & nn/ML & NN lab (complete)/datasets/breast cancer_week 8 and 9.csv')
data


# In[13]:


#read the data set 
#option 2
path="C:/Users/saile/Desktop/ml & nn/ML & NN lab (complete)/datasets/breast cancer_week 8 and 9.csv"
data=pd.read_csv(path,encoding='utf-8')
data


# In[14]:


data.shape


# In[15]:


data.ndim


# In[16]:


data.head()


# In[17]:


data.tail()


# In[18]:


data.head(10)


# In[22]:


print(data[['id','radius_mean']])


# In[25]:


data['id'].describe()


# In[27]:


data.describe()


# In[24]:


data.dtypes


# In[37]:


data.columns


# In[33]:


my_list=list(data)
print(my_list)
print(type(my_list))


# In[34]:


print(data['radius_mean'])


# In[35]:


data.radius_mean


# In[36]:


data.iloc[:,2]


# In[38]:


# Rename columns using a dictionary to map values
# Rename the id columnn to 'serialno'
data = data.rename(columns={"id": "serialno"})

# Again, the inplace parameter will change the dataframe without assignment
data.rename(columns={"id": "serialno"}, inplace=True)


data


# In[41]:


# Rename multiple columns in one go with a larger dictionary
data.rename(
    columns={
        "serialno": "serial_no",
        "diagnosis": "m/b"
    },
    inplace=True
)

# Rename all columns using a function, e.g. convert all column names to lower case:
data.rename(columns=str.lower)


# In[42]:


data_cbind_1 = pd.DataFrame({"x1":range(10, 16),                   # Create first pandas DataFrame
                             "x2":range(30, 24, - 1),
                             "x3":["a", "b", "c", "d", "e", "f"],
                             "x4":range(48, 42, - 1)})
print(data_cbind_1)                                                # Print first pandas DataFrame


# In[43]:


data_cbind_2 = pd.DataFrame({"y1":["foo", "bar", "bar", "foo", "foo", "bar"], # Create second pandas DataFrame
                             "y2":["x", "y", "z", "x", "y", "z"],
                             "y3":range(18, 0, - 3)})
print(data_cbind_2)                                                # Print second pandas DataFrame


# In[44]:


#By executing the previous code, we have managed to construct Tables 1 and 2, i.e. two pandas DataFrames.

#Note that these two data sets have the same number of rows. This is important when applying a column-bind.

#In the next step, we can apply the concat function to column-bind our two DataFrames:

data_cbind_all = pd.concat([data_cbind_1.reset_index(drop = True), # Cbind DataFrames
                            data_cbind_2],
                           axis = 1)
print(data_cbind_all)                                              # Print combined DataFrame


# In[45]:


#Example 2: Column-bind Two pandas DataFrames Using ID Column

data_merge_1 = pd.DataFrame({"ID":range(1, 5),                     # Create first pandas DataFrame
                             "x1":range(10, 14),
                             "x2":range(30, 26, - 1),
                             "x3":["a", "b", "c", "d"],
                             "x4":range(48, 44, - 1)})
print(data_merge_1)      


# In[46]:


data_merge_2 = pd.DataFrame({"ID":range(3, 9),                     # Create second pandas DataFrame
                             "y1":["foo", "bar", "bar", "foo", "foo", "bar"],
                             "y2":["x", "y", "z", "x", "y", "z"],
                             "y3":range(18, 0, - 3)})
print(data_merge_2)                                                # Print second pandas DataFrame


# In[47]:


data_merge_all = pd.merge(data_merge_1,                            # Cbind DataFrames
                          data_merge_2,
                          on = "ID",
                          how = "outer")
print(data_merge_all)                                              # Print combined DataFrame


# In[50]:


#Example 3: Combine pandas DataFrames rowwise
data_rbind_1 = pd.DataFrame({"x1":range(11, 16),                   # Create first pandas DataFrame
                             "x2":["a", "b", "c", "d", "e"],
                             "x3":range(30, 25, - 1),
                             "x4":range(30, 20, - 2)})
print(data_rbind_1)                                                # Print first pandas DataFrame


# In[51]:


data_rbind_2 = pd.DataFrame({"x1":range(3, 10),                    # Create second pandas DataFrame
                             "x2":["x", "y", "y", "y", "x", "x", "y"],
                             "x3":range(20, 6, - 2),
                             "x4":range(28, 21, - 1)})
print(data_rbind_2)                                                # Print second pandas DataFrame







# In[52]:


data_rbind_all = pd.concat([data_rbind_1, data_rbind_2],           # Rbind DataFrames
                           ignore_index = True,
                           sort = False)
print(data_rbind_all)  


# In[53]:


data.isnull()


# In[55]:


data.isnull().sum()


# In[59]:


# Output data to a CSV file
# Typically, I don't want row numbers in my output file, hence index=False.
# To avoid character issues, I typically use utf8 encoding for input/output.

data.to_csv("C:/Users/saile/Desktop/output_filename.csv", index=False, encoding='utf8')

# Output data to an Excel file.
# For the excel output to work, you may need to install the "xlsxwriter" package.

data.to_csv("C:/Users/saile/Desktop/output_excel_file.xlsx", index=False)


# In[ ]:




