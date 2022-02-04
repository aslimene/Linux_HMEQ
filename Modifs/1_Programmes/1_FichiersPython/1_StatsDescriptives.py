#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
#import pandas_profiling


# In[44]:


# import data
data = pd.read_csv('hmeq.csv',index_col=False)


# In[5]:


data.dtypes.value_counts()


# In[6]:


data.dtypes.value_counts().plot.pie()


# - La plupart des données sont numériques, seules deux sont de type chaînes (REASON et JOB).

# In[45]:


data.describe()


# ### Analyse univariate

# In[34]:


data['BAD'].value_counts()


# In[46]:


data['REASON'].value_counts(normalize = True)


# In[48]:


100*data['JOB'].value_counts(normalize = True)


# # VISUAL EDA (Explanatory Data Analysis) Categorical

# In[36]:


sns.countplot(x='BAD', data = data)


# In[13]:


sns.catplot(x='BAD', col = 'REASON',kind='count', data=data)


# In[43]:


d = data[data["REASON"]=="HomeImp"]
d.BAD.value_counts(normalize=True)


# In[14]:


plt.figure(figsize = (20, 12))
sns.catplot(x='BAD', col = 'JOB',kind='count', data=data)


# # VISUAL EDA Numerical

# In[15]:


sns.lmplot(x='YOJ', y='LOAN', hue='BAD', data=data, fit_reg=False, scatter_kws={'alpha':0.5});


# In[16]:


plt.figure()
sns.lmplot(x='VALUE', y='LOAN', hue='BAD', data=data, fit_reg=False, scatter_kws={'alpha':0.5})
plt.show()


# In[32]:


plt.figure(figsize = (10, 6))
sns.heatmap(data.corr(), cmap="Blues")

