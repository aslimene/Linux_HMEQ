#!/usr/bin/env python
# coding: utf-8

# In[70]:


#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
import warnings
warnings.filterwarnings("ignore")

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


selectors = [RFE, SequentialFeatureSelector, ExhaustiveFeatureSelector]
algos = [LogisticRegression()]
recaps = selection_list(selectors, algos, X_train, y_train, X_test, metric = "roc_auc", smote = True)
recaps


# In[42]:


list_vars = recaps.iloc[0, 5]
selector = recaps.iloc[0, 0]
estimateur = recaps.iloc[0, 1]

nb_vars = recaps.iloc[0, 3]
X_train = X_train[list_vars]
X_test = X_test[list_vars]
#print("X_train shape :", X_train.shape)
print("L'estimateur", estimateur, "avec la méthode", selector, "a selectionné", nb_vars, "variables :", list_vars)


# In[19]:


val(X_train, y_train, smote = False)


# In[43]:


#from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_score

overfit(['roc_auc', 'accuracy', "recall", 'precision', 'f1'], X_train, y_train ).round(3)


# In[22]:


scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 
           'Recall': make_scorer(recall_score), 'f1_score': make_scorer(f1_score),
          'Precision': make_scorer(precision_score)}

param={"penalty":["l2", "l1", "elasticnet"], 
       'C':[5,10,15,20], 
       "max_iter":[100,150,200]}

gs = GridSearchCV(LogisticRegression(random_state=42),
                  param_grid=param,
                  scoring=scoring, refit='AUC', return_train_score=True, cv = 5)

gs.fit(X_train, y_train)
results = gs.cv_results_

print("Les meilleurs paramètres sont:", gs.best_params_)
print("Le meilleur score est: %.2f" %gs.best_score_)


# In[40]:


train_pred, y_test, test_pred, fpr, tpr, roc_auc = model_perf(X_train, y_train, X_test, y_test, smote = True, show_roc= True, show_prc=True, cut_off= 0.45, show_conf_matrix=True)


# In[74]:


def scoring(X, base = 1000, pdo = 30):
    
    #CREATION DE LA GRILLE SCORE À PARTIR DES CLASSES, DU MODÈLE M ET CALIBRAGE SUR "base" POINTS
    lr = LogisticRegression(penalty='l2', C=5, max_iter= 100, random_state=42)
    lr.fit(X_train, y_train)
    test_pred = lr.predict_proba(X_test)[:,1]
    
    ppb = sc.scorecard(classes, lr, xcolumns=X_test.columns, 
                       odds0=1/500,  points0=base, pdo=pdo, basepoints_eq0 = True)
    
    
    #CALCUL DES SCORES TOTAUX DANS LE TRAIN SET
    train_score = sc.scorecard_ply(train, ppb, print_step=0, only_total_score=False)
        #CALCUL DES SCORES DANS LE TEST SET
    test_score = sc.scorecard_ply(test, ppb, print_step=0, only_total_score=False)

    if X == "Test":
        a = test_score
        b = test
    else:
        X == "Train"
        a = train_score
        b = train
        
    score_avec_target = pd.concat([a, b['BAD']],axis=1)
    #score_total       = score_avec_target['score']
    bon_score         = score_avec_target[score_avec_target["BAD"]== 0]['score']
    mauvais_score     = score_avec_target[score_avec_target["BAD"]== 1]['score']


    plt.figure()
    sns.distplot(mauvais_score, color='red', label = "mauvais score")
    sns.distplot(bon_score, color = "green", label ="bon score" )
    plt.legend()
    plt.title("score distribution - " + str(X))

        
    plt.figure()
    sns.distplot(train_score["score"], color='blue', label = "train score")
    sns.distplot(test_score["score"], color = "yellow", label ="test score" )
    plt.legend()
    plt.title("Test & Train Comparaison")
    
    return ppb, train_score, test_score

ppb, train_score, test_score= scoring("Test")
ppb, train_score, test_score= scoring("Train")


# In[46]:


test_score.describe()["score"]


# In[47]:


train_score.describe()["score"]


# In[48]:


test_score.head()


# In[49]:


ppb


# In[50]:


ppb['VALUE']


# In[51]:


ppb['JOB']


# In[52]:


a = sc.perf_psi(score = {'train':train_score, 'test':test_score}, 
        label = {'train':y_train, 'test':y_test},
        return_distr_dat=True)


# In[53]:


a.keys()


# In[54]:


a["psi"]


# In[55]:


a["dat"]


# In[56]:


a["dat"]["CLAGE_points"]


# In[57]:


a["dat"]["score"]


# In[ ]:





# In[58]:


test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['ks'])


# In[59]:


plt.figure(figsize=(10, 6))
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['roc'])


# In[60]:


test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['lift'])


# In[61]:


test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['pr'])

