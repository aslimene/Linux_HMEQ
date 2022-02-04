#!/usr/bin/env python
# coding: utf-8

# # ***Abdoul Aziz BERRADA - Amira SLIMENE***                                  
#                                                                                

# ## ***Scoring et Risque de défaut***
# 
# ### ***Scorecard building using HMEQ data***

# In[109]:


import warnings
warnings.filterwarnings("ignore")
import time
import re

import utils

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import skew
import scorecardpy as sc
import statsmodels.api as sm
from statsmodels.tools import tools
from statsmodels.iolib.summary import Summary
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.metrics import make_scorer, auc, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
import pandas_profiling


# ## **1 - EXPLORATORY DATA EXPLORATION**

# In[90]:


data.dtypes.value_counts()


# In[91]:


data.dtypes.value_counts().plot.pie()


# ### Analyse univariate

# In[100]:


data.describe()


# In[93]:


data['BAD'].value_counts()


# In[94]:


data['REASON'].value_counts()


# In[95]:


sns.countplot(x='BAD', data = data)


# ### Analyse multiivariate

# In[96]:


sns.catplot(x='BAD', col = 'REASON',kind='count', data=data)


# In[97]:


plt.figure(figsize = (20, 12))
sns.catplot(x='BAD', col = 'JOB',kind='count', data=data)


# In[98]:


sns.lmplot(x='YOJ', y='LOAN', hue='BAD', data=data, fit_reg=False, scatter_kws={'alpha':0.5})


# In[99]:


plt.figure()
sns.lmplot(x='VALUE', y='LOAN', hue='BAD', data=data, fit_reg=False, scatter_kws={'alpha':0.5})
plt.show()


# In[101]:


plt.figure(figsize = (10, 6))
sns.heatmap(data.corr(), cmap="Blues")


# ## **2 - DATA CLEANING**

# ###  **Données manquantes**

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

# In[5]:


train, test = train_test_split(data, test_size = 0.2, random_state = 42)


# In[8]:


data.shape


# In[9]:


train.shape


# In[10]:


train.columns


# In[6]:


prct_num, prct_qual = utils.prct_nan(train)


# In[7]:


vars_num = [train.select_dtypes('float').columns]
vars_qual = [train.select_dtypes('object').columns]


# In[8]:


prct_num = [print((100*train[col].isna().sum()/train.shape[0]).sort_values(ascending = False)) for col in vars_num]


# In[9]:


prct_qual = [print((100*train[col].isna().sum()/train.shape[0]).sort_values(ascending = False)) for col in vars_qual]


# In[10]:


prct_num, prct_qual = utils.prct_nan(train)


# In[11]:


prct_num, prct_qual = utils.prct_nan(test)


# In[12]:


train, test = utils.imputation(train, test)


# In[13]:


data_clean = pd.concat([train.reset_index(drop=True), test.reset_index(drop=True)], axis= 0)


# In[14]:


data_clean.head()


# ## **3 - FEATURES SELECTION**

# ###    **3-1 WOE BINNING & IV**

# In[15]:


classes = sc.woebin(data_clean, y="BAD", positive=1, method="chimerge")


# In[16]:


sc.woebin_plot(classes)


# In[23]:


classes["LOAN"]


# In[17]:


classes["VALUE"]


# In[18]:


train_woe = sc.woebin_ply(train, classes)
test_woe = sc.woebin_ply(test, classes)


# In[19]:


train_woe.head()


# In[20]:


y_train = train_woe["BAD"]
X_train = train_woe.drop(["BAD",'REASON_woe' ], axis =1)

y_test = test_woe["BAD"]
X_test = test_woe.drop(["BAD", 'REASON_woe'], axis =1)


# In[21]:


print("Les features sélectionnées sont:")
print(X_train.columns)
print("Elles sont au nombre de", X_train.shape[1])


# ###    **3-2 AUTOMATIC SELECTION**

# In[25]:


all_scores_dict = utils.selection_np(RFE, LogisticRegression(), X_train, y_train, X_test, Forward = False, metric='roc_auc')
all_scores_dict


# In[30]:


selectors = [RFE, SequentialFeatureSelector, ExhaustiveFeatureSelector]
algos = [LogisticRegression(random_state=42)]
recaps = utils.selection_list(selectors, algos, X_train, y_train, X_test, metric = "roc_auc", smote = True)
recaps


# In[31]:


list_vars = recaps.iloc[2, 5]
selector = recaps.iloc[2, 0]
estimateur = recaps.iloc[2, 1]

nb_vars = recaps.iloc[2, 3]
X_train = X_train[list_vars]
X_test = X_test[list_vars]
#print("X_train shape :", X_train.shape)
print("L'estimateur", estimateur, "avec la méthode", selector, "a selectionné", nb_vars, "variables :", list_vars)


# ### ***3-3 SIGNIFICATIVITE NON NULLE***

# In[29]:


utils.val(X_train, y_train)


# ## ***4 - MODEL SELECTION - PARAMETERS SELECTION***

# In[32]:


utils.overfit(['roc_auc', 'accuracy', "recall", 'precision', 'f1'], X_train, y_train ).round(3)


# In[33]:


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


# In[34]:


train_pred, y_test, test_pred, fpr, tpr, roc_auc = utils.model_perf(X_train, y_train, X_test, y_test, smote = True, show_roc= True, show_prc=True, cut_off= 0.45, show_conf_matrix=True)


# ## ***5 - SCORECARD***

# In[88]:


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
    sns.distplot(mauvais_score, color='red', label = "BAD = 1")
    sns.distplot(bon_score, color = "green", label ="BAD = 0" )
    plt.legend(loc='upper left')
    plt.title("score distribution - " + str(X))

        
    plt.figure()
    sns.distplot(train_score["score"], color='blue', label = "train score")
    sns.distplot(test_score["score"], color = "orange", label ="test score" )
    plt.legend(loc='upper left')
    plt.title("Test & Train Comparaison")
    
    return ppb, train_score, test_score


# In[89]:


ppb, train_score, test_score= scoring("Test")
ppb, train_score, test_score= scoring("Train")


# In[38]:


test_score.describe()["score"]


# In[39]:


train_score.describe()["score"]


# In[40]:


test_score.head()


# In[41]:


ppb


# In[42]:


ppb['VALUE']


# In[44]:


p = sc.perf_psi(score = {'train':train_score, 'test':test_score}, 
        label = {'train':y_train, 'test':y_test},
        return_distr_dat=True)


# In[80]:


p.keys()


# In[79]:


p["psi"]


# In[45]:


scorecard_bins = p["dat"]["score"]
scorecard_bins


# In[46]:


test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['ks'])
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['lift'])
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['roc'])
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['pr'])


# ## ***6 - MODEL CHALLENGING***

# In[48]:


utils.grid_search("RandomForestClassifier",  X_train, y_train )


# In[50]:


utils.grid_search("DecisionTreeClassifier",  X_train, y_train )


# In[51]:


utils.grid_search("LinearDiscriminantAnalysis",  X_train, y_train )


# In[52]:


utils.grid_search("GradientBoostingClassifier",  X_train, y_train )


# In[53]:


utils.grid_search("KNeighborsClassifier", X_train, y_train )


# In[55]:


def model_comp(model, X_train = X_train, y_train = y_train, 
          X_test = X_test, y_test = y_test, 
          smote = True, show_roc = True, show_conf_matrix = False, cut_off = 0.45, show_prc = False) : 
    
    b = re.findall(r"[A-Za-z0-9]+", str(model))
    if smote == True:
        print("SMOTE")
        print("--------")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42 , k_neighbors=15)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    lr = model.fit(X_train, y_train)
    print("train score %.2f" %lr.score(X_train, y_train))
    print("--------")
    
        #PREDICTION DES Y_PRED
    train_pred = lr.predict_proba(X_train)[:,1]
    test_pred = lr.predict_proba(X_test)[:,1]
    print("test score %.2f" %lr.score(X_test, y_test))
    
    print("--------")
    print("Cut-off : ", cut_off)
    
        #ACCURACY, PRECISION ET TOUTE LA CLIQUE AU SEUIL DE 50% DE CONFIANCE (STANDARD)
    print("--------")
    print("classification_report\n")
    test_pred_b = (test_pred > cut_off).astype(bool)
    train_pred_b = (train_pred > cut_off).astype(bool)
    print(classification_report(y_test, test_pred_b))
    print("\n")

    train_pred_b = (train_pred > cut_off).astype(bool)
    
            #MATRICE DE CONFUSION
    if show_conf_matrix == True : 
        print("Matrice de confusion\n")
        print(confusion_matrix(y_test, test_pred_b))
        print("--------")
        #AUC
    fpr, tpr, _= roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)
    fpr_, tpr_, _train= roc_curve(y_train, train_pred)
    roc_auc_ = auc(fpr_, tpr_)
    print("L'AUC est de %.2f" %roc_auc)
    print("--------")
    Gini = roc_auc * 2 - 1
    print('Gini: %.2f' % (Gini))
    print("--------")
    
    if show_roc == True:
        print('roc curve')
        
        plt.figure(figsize=(6,6))
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.plot(fpr, tpr, lw=3, label='roc curve - Test (auc = {:0.2f})'.format(roc_auc))
        #plt.plot(fpr_, tpr_, lw=3, label='roc curve - Train (auc = {:0.2f})'.format(roc_auc_))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(b[0], fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.axes().set_aspect('equal')
        plt.show()
        print("\n")
        
       
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred)
    precision_, recall_, thresholds_ = precision_recall_curve(y_train, train_pred)
    
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    
    closest_zero_ = np.argmin(np.abs(thresholds_))
    closest_zero_p_ = precision_[closest_zero_]
    closest_zero_r_ = recall_[closest_zero_]
    
    if show_prc == True : 
        
        print("precision_recall_curve")
        plt.figure(figsize = (6,6))
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.plot(precision, recall, label='prc - Test')
        #plt.plot(precision_, recall_, label='prc - Train')
        #plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
        #plt.plot(closest_zero_p_, closest_zero_r_, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
        plt.xlabel('Precision', fontsize=16)
        plt.ylabel('Recall', fontsize=16)
        plt.axes().set_aspect('equal')
        plt.legend(loc='lower left', fontsize=13)
        plt.show()

    J = tpr - fpr
    ix = np.argmax(J)
    seuil_opt = thresholds[ix]
    
    J_ = tpr_ - fpr_
    ix_ = np.argmax(J_)
    seuil_opt_ = thresholds_[ix_]
    print('Meilleur cut-off test : %f' % (seuil_opt))
    print('Meilleur cut-off train : %f' % (seuil_opt_))
    
    return fpr, tpr, roc_auc


# In[56]:


fpr_r, tpr_r, roc_auc_r = model_comp(model = RandomForestClassifier(max_depth=100, max_features=4, n_estimators = 25, random_state=42))


# In[72]:


fpr_d, tpr_d, roc_auc_d = model_comp(model = DecisionTreeClassifier(max_depth=4, max_features=10))


# In[58]:


fpr_gb, tpr_gb, roc_auc_gb = model_comp(model = GradientBoostingClassifier(learning_rate=0.2, max_depth=1, n_estimators=25, random_state= 42))


# In[59]:


fpr_lda, tpr_lda, roc_auc_lda = model_comp(model = LinearDiscriminantAnalysis())


# In[73]:


fpr_knn, tpr_knn, roc_auc_knn = model_comp(model = KNeighborsClassifier(n_neighbors=35))


# In[77]:


clf1 = KNeighborsClassifier(n_neighbors=35)
clf2 = RandomForestClassifier(max_depth=75, max_features=3, n_estimators = 25, random_state=42)
clf3 = LogisticRegression(penalty='l2', C=15, max_iter= 100, random_state=42)
clf4 = LinearDiscriminantAnalysis()
clf5 = DecisionTreeClassifier(max_depth=8, max_features=9)
clf6 = GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=25, random_state=2)


estimators=[('knn', clf1), ('rf', clf2), ('lr', clf3), ('lda', clf4), ('dt', clf5), ('gb', clf6)]
vclf = VotingClassifier(estimators, voting='soft')
fpr_vc, tpr_vc, roc_auc_vc = model_comp(model = vclf)


# In[78]:


sclf = StackingClassifier(estimators)
fpr_sc, tpr_sc, roc_auc_sc = model_comp(model = sclf)


# In[79]:


plt.figure(figsize=(6,6))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_r, tpr_r, lw=3, label='RF (auc = {:0.2f})'.format(roc_auc_r))
plt.plot(fpr_d, tpr_d, lw=3, label='DT (auc = {:0.2f})'.format(roc_auc_d))
plt.plot(fpr_gb, tpr_gb, lw=3, label='GB (auc = {:0.2f})'.format(roc_auc_gb))
plt.plot(fpr_lda, tpr_lda, lw=3, label='LDA (auc = {:0.2f})'.format(roc_auc_lda))
plt.plot(fpr_knn, tpr_knn, lw=3, label='LR (auc = {:0.2f})'.format(roc_auc_knn))

plt.plot(fpr_sc, tpr_sc, lw=3, label='SK (auc = {:0.2f})'.format(roc_auc_sc))
plt.plot(fpr_vc, tpr_vc, lw=3, label='LR (auc = {:0.2f})'.format(roc_auc_vc))
plt.plot(fpr, tpr, lw=3, label='LR (auc = {:0.2f})'.format(roc_auc))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ML Models', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# 
