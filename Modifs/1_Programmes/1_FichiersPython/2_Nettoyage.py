#!/usr/bin/env python
# coding: utf-8

# In[22]:


#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
import warnings
warnings.filterwarnings("ignore")

from utils import *


# In[2]:


data = pd.read_csv("hmeq.csv")
data.head(5)


# In[3]:


data.info()


# In[4]:


#A supprimer après

print(data.BAD.value_counts())
print(data.BAD.value_counts(normalize = True))


# In[5]:


plt.figure(figsize=(10,6))
sns.countplot(data=data, x='BAD')


#  ## **Données manquantes**

# 
# La méthodologie employée pour imputer les données manquantes est la suivante : 
# 
# RQ : On considère qu'au delà de 40% de valeurs manquantes, une stratégie d'imputation risque d'introduire un biais dans l'analyse donc on va pas utiliser une telle variable par la suite.
# 
# - Variables quantitatives : 
# On va se baser sur la distribution de la variable, si sa distribution est asymétrique, on va imputer par la médiane et sinon par la moyenne.
# - Variables qualitatives : 
# On va considérer le % de données manquantes, si celui-ci est inférieur à 15%, on va imputer par la valeur la plus fréquente (donc le mode), sinon on sera amené à créer une nouvelle classe nommée "autres". 
# 
# Les valeurs imputées dans le train_set seront repercutées dans le test_set

# In[6]:


train, test = train_test_split(data, test_size = 0.2, random_state = 42)


# In[7]:


data.shape


# In[8]:


train.shape


# In[9]:


train.columns


# In[10]:


test.shape


# In[11]:


prct_num, prct_qual = prct_nan(train)


# In[12]:


vars_num = [train.select_dtypes('float').columns]
vars_qual = [train.select_dtypes('object').columns]


# In[13]:


prct_num = [print((100*train[col].isna().sum()/train.shape[0]).sort_values(ascending = False)) for col in vars_num]


# In[14]:


prct_qual = [print((100*train[col].isna().sum()/train.shape[0]).sort_values(ascending = False)) for col in vars_qual]


# In[15]:


prct_num, prct_qual = prct_nan(train)


# In[16]:


prct_num, prct_qual = prct_nan(test)


# In[17]:


train, test = imputation(train, test)


# In[18]:


data_clean = pd.concat([train.reset_index(drop=True), test.reset_index(drop=True)], axis= 0)


# In[19]:


data_clean.head()


# In[20]:


data_clean.shape

