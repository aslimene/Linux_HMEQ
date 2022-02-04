#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import time

from utils import *


# In[2]:


path="/Users/Abdoul_Aziz_Berrada/Documents/M2_MOSEF/2_Projets/Semestre1/CreditScoring/"
train = pd.read_csv(path+"train.csv")
train = train.drop("Unnamed: 0", axis = 1)
test = pd.read_csv(path+"test.csv")
test = test.drop("Unnamed: 0", axis = 1)
data = pd.read_csv(path+"data_clean.csv")
data = data.drop("Unnamed: 0", axis = 1)


# In[3]:


test.shape


# In[4]:


train.shape


# In[5]:


data.shape


# In[6]:


data.head()


# In[7]:


import scorecardpy as sc


# In[8]:


classes = sc.woebin(data, y="BAD", positive=1, method="chimerge")


# In[9]:


sc.woebin_plot(classes)


# In[10]:


classes["LOAN"]


# In[11]:


classes["VALUE"]


# In[12]:


train_woe = sc.woebin_ply(train, classes)
test_woe = sc.woebin_ply(test, classes)


# In[13]:


train_woe.head()


# In[14]:


y_train = train_woe["BAD"]
X_train = train_woe.drop(["BAD",'REASON_woe' ], axis =1)

y_test = test_woe["BAD"]
X_test = test_woe.drop(["BAD", 'REASON_woe'], axis =1)


# In[15]:


print("Les features sélectionnées sont:")
print(X_train.columns)
print("Elles sont au nombre de", X_train.shape[1])


# In[16]:


all_scores_dict = selection_np(RFE, LogisticRegression(), X_train, y_train, X_test, Forward = False, metric='roc_auc')
all_scores_dict


# In[18]:


selectors = [RFE, SequentialFeatureSelector, ExhaustiveFeatureSelector]
algos = [LogisticRegression()]
recaps = selection_list(selectors, algos, X_train, y_train, X_test, metric = "roc_auc", smote = True)
recaps


# In[19]:


list_vars = recaps.iloc[0, 5]
selector = recaps.iloc[0, 0]
estimateur = recaps.iloc[0, 1]

nb_vars = recaps.iloc[0, 3]
X_train = X_train[list_vars]
X_test = X_test[list_vars]
print("L'estimateur", estimateur, "avec la méthode", selector, "a selectionné", nb_vars, "variables :", list_vars)


# In[20]:


val(X_train, y_train)

