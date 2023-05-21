#!/usr/bin/env python
# coding: utf-8

# Week 2:
# Basic statistical functions for data exploration
# 1. Measures of central tendency – mean, median, mode
# 2. Measures of data spread
# 3. Dispersion of data – variance, standard deviation
# 4. Position of the different data values – quartiles, inter-quartile range (IQR).

# In[1]:


#to calculate mean
from collections import Counter
numb = [2, 3, 5, 7, 8,2,2,3,4,5,2]
no = len(numb)
summ = sum(numb)
mean = summ / no
print("The mean or average of all these numbers (", numb, ") is", str(mean))

# to calculate median
no = len(numb)
numb.sort()
if no % 2 == 0:
    median1 = numb[no//2]
    median2 = numb[no//2 - 1]
    median = (median1 + median2)/2
else:
    median = numb[no//2]
print("The median of the given numbers  (", numb, ") is", str(median))

# to calulate  mode
no = len(numb)
val = Counter(numb)
findMode = dict(val)
mode = [i for i, v in findMode.items() if v == max(list(val.values()))]  
if len(mode) == no:
    findMode = "The group of number do not have any mode"
else:
    findMode = "The mode of a number is / are: " + ', '.join(map(str, mode))
print(findMode)


# In[2]:


#quartile and interquartile

data = sorted(list(map(int,input("Input numbers with space > ").split())))
n = len(data)
i = n // 2

if n % 2 == 0:
	median = (data[i-1] + data[i])/2
	q3i = 0
else:
	median = data[i]
	q3i = 0

nquartile = n // 2
i = nquartile // 2

if nquartile % 2 == 0:
	q1 = (data[i-1] + data[i])/2
	q3 = (data[q3i + nquartile + i - 1] + data[q3i + nquartile + i]) / 2
else:
	q1 = data[i]
	q3 = data[q3i + nquartile + i]

print(data)
print("Q1 =", q1)
print("Q2 =", median, "(median)")
print("Q3 =", q3)
print("Interquartile =", q3 - q1)


# In[9]:



#standard deviation

from math import sqrt
n= [11, 8, 8, 3, 4, 4, 5, 6, 6, 7, 8] 
mean =sum(n)/len(n)
SUM= 0
for i in n :
    SUM +=(i-mean)**2
stdeV = sqrt(SUM/(len(n)-1)) 
print(stdeV)


# In[10]:


#variance
#define a function, to calculate variance
def variance(X):
    mean = sum(X)/len(X)
    tot = 0.0
    for x in X:
        tot = tot + (x - mean)**2
    return tot/len(X)
 
# call the function with data set
x = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
print("variance is: ", variance(x))
 
y = [1, 2, 3, -4, -5, -6, -7, -8] 
print("variance is: ", variance(y))
 
z = [10, -20, 30, -40, 50, -60, 70, -80] 
print("variance is: ", variance(z))


# In[2]:


import pandas as pd
import numpy as np
import statistics as st 

# Load the data
df = pd.read_csv("C:/Users/saile/Desktop/week2.csv")
print(df.shape)


# In[3]:


df.mean()


# In[4]:


df.std()


# In[5]:


df.var()


# In[6]:


from scipy.stats import iqr


# 
