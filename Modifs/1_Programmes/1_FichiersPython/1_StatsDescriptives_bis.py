#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import pandas_profiling


# In[21]:


data = pd.read_csv('hmeq.csv',index_col=False)


# In[27]:


profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True, dark_mode=True)
profile

