#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:59:06 2021

@author: Abdoul_Aziz_Berrada
"""

#------------
import time
import re

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.metrics import make_scorer, auc, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE


#-----------


def prct_nan(x): 
    """
    Parametre
    ----------
    x = a DataFrame 
    
    Returns
    --------

    Cette fonction permet de déterminer le pourcentage de valeurs manquantes des variables d'un DataFrame.
    Elle calcule séparément les variables qualitatives et quantitatives.
    
    Example
    --------
    >>> import pandas pd
    >>> x = pd.read_csv("dataframe.csv")

    >>> prct_num, prct_qual = prct_nan(x)
    
    """
    vars_num = [x.select_dtypes('float').columns]
    vars_qual = [x.select_dtypes('object').columns]
    
    print("Pour les variables numériques, on a : \n")
    prct_num = [print((100*x[col].isna().sum()/x.shape[0]).sort_values(ascending = False)) for col in vars_num]
    print("-----")
    print("Pour les variables quantitatives, on a : \n")
    prct_qual = [print((100*x[col].isna().sum()/x.shape[0]).sort_values(ascending = False)) for col in vars_qual]
    return prct_num, prct_qual


def imputation(x1, x2):
    """
    Parametres
    ----------
    x1 = DataFrame, qui doit être le Train set
    x2 = DataFrame, qui doit être le Test set
    
    Returns
    --------
    Cette fonction permet d'imputer les valeurs manquantes selon la méthode définie dans la méthodologie.
    Elle retourne 2 parties complémentaires d'un DataFrame après imputation des valeurs manquantes.
    
    Example
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> from scipy.stats import skew
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    >>> X_train, X_test = imputation(X_train, X_test)
    """
    from sklearn.impute import SimpleImputer
    from scipy.stats import skew
    
    #Colonnes numériques
        #train
    col_asy_tr = []
    col_sym_tr = []
        #test
    col_asy_ts = []
    col_sym_ts = []
    #Colonnes qualitatives
        #train
    col_qual_tr = []
    col_qual_tr_ = []
        #test
    col_qual_ts = []
    col_qual_ts_ = []

    for col in x1.select_dtypes("float").columns:
        if skew(x1[col], nan_policy='omit').data > 1 : 
            col_asy_tr.append(col)
        else : 
            col_sym_tr.append(col)

    for col in x2.select_dtypes("float").columns:
        if skew(x2[col], nan_policy='omit').data > 1 : 
            col_asy_ts.append(col)
        else : 
            col_sym_ts.append(col)
            
    print("AVANT IMPUTATION\n\n")
    print("**Variables quantitatives**:\n")
    print("---Train set----")
    print("Colonnes asymétriques dans le train", col_asy_tr)
    print("Colonnes symétriques dans le train", col_sym_tr)
    print("----Test set----")
    print("Colonnes asymétriques dans le test", col_asy_ts)
    print("Colonnes symétriques dans le test", col_sym_ts)

    for col in x1.select_dtypes("object").columns:
        if np.any(x1.isna().sum()/x1.shape[0]) > 0.15 : 
            col_qual_tr.append(col)
        else : 
            col_qual_tr_.append(col)

    for col in x2.select_dtypes("object").columns:
        if np.any(x2.isna().sum()/x2.shape[0]) > 0.15 : 
            col_qual_ts.append(col)
        else : 
            col_qual_ts_.append(col)
    print("\n")
    print("**Variables qualitatives**:\n")
    print("---Train set----")
    print("Colonnes avec - de 15% de nan dans le train", col_qual_tr)
    print("Colonnes avec + de 15% de nan dans le train", col_qual_tr_)
    print("----Test set----")
    print("Colonnes avec - de 15% de nan dans le test", col_qual_ts)
    print("Colonnes avec + de 15% de nan dans le test", col_qual_ts_)
    
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    imp_median = SimpleImputer(missing_values=np.nan,strategy='median')
    imp_mode = SimpleImputer(missing_values=np.nan,strategy='most_frequent')


    x1[col_sym_tr] = imp_mean.fit_transform(x1[col_sym_tr])
    x1[col_asy_tr] = imp_median.fit_transform(x1[col_asy_tr])
    x1[col_qual_tr] = imp_mode.fit_transform(x1[col_qual_tr])

    x2[col_sym_ts] = imp_mean.fit_transform(x2[col_sym_ts])
    x2[col_asy_ts] = imp_median.fit_transform(x2[col_asy_ts])
    x2[col_qual_ts] = imp_mode.fit_transform(x2[col_qual_ts])
    
    print("\n\n")
    print("APRÉS IMPUTATION")
    print("Valeurs manquantes dans le test set\n")
    print(x2.isna().sum())
    print("\n")
    print("Valeurs manquantes dans le train set\n")
    print(x1.isna().sum())
    return x1, x2



def selection_np(selector, estimateur, X_train, y_train, X_test, Forward = False, metric = "accuracy", 
                 smote = False, cv = 5):  
 
    """
    Cette fonction affiche pour chaque selector et estimateur, le nombre et les variables selectionnées.
    Si le selector est la RFE ou ExhaustiveFeatureSelector, la fonction affiche le score obtenu après la sélection.
    
    Paramètres :
    ------
    selector : {"RFE", "SequentialFeatureSelector", "ExhaustiveFeatureSelector"}
            Une méthode de sélection de variables,
                
    estimateur : Tout algorithme de classification type sklearn,
    
    Forward : bool (default: True)
            Forward selection si True,
    
    print_estimateur : bool (default: False)
                    Affiche le nom de l'estimateur utilisé
    
    Note
    ------
    Nous recommandons de ne pas utiliser une technique de Validation croisée avec la selection ExhaustiveFeatureSelector.
    En effet elle l'inclut déjà.
    
    Examples
    ------
    >>>from sklearn.linear_model import LogisticRegression
    >>>from sklearn.feature_selection import RFE
    
    selection(RFE, LogisticRegression(), print_estimateur = True)
    """
    import time
    import re
    
    if smote == True:


        smote = SMOTE(random_state=0 ,k_neighbors=15)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if selector == RFE:

        a = re.findall(r"\w+", str(selector))
        t1 = time.time()
        nov_list=np.arange(1, X_train.shape[1]+1)            
        high_score=0
        nov=0           
        score_list =[]
        for n in range(len(nov_list)):
            model = estimateur
            b = re.findall(r"[A-Za-z0-9\(\)\_\=\,]+", str(model))
            sel = selector(model,nov_list[n])
            X_train_sel = sel.fit_transform(X_train, y_train)
            X_test_sel = sel.transform(X_test)
            model.fit(X_train_sel,y_train)
            scores = cross_val_score(model, X_train_sel, y_train, scoring = metric, cv = cv)
            score = np.mean(scores)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nov = nov_list[n]
        t2 = time.time() - t1
        temp = pd.Series(sel.support_,index = X_train.columns)
        variables = temp[temp==True].index

    elif selector == SequentialFeatureSelector:
        t1 = time.time()
        a = re.findall(r"\w+", str(selector))
        model = estimateur
        b = re.findall(r"[A-Za-z0-9\(\)\_\=\,]+", str(model))
        sfs = SequentialFeatureSelector(model, k_features = "best", forward = Forward, scoring = metric, cv = cv )
        sfs.fit(X_train, y_train)
        variables = list(sfs.k_feature_names_)
        t2 = time.time() - t1
        high_score = sfs.k_score_

    elif selector == ExhaustiveFeatureSelector:
        a = re.findall(r"\w+", str(selector))
        t1 = time.time()
        model = estimateur
        b = re.findall(r"[A-Za-z0-9\(\)\_\=\,]+", str(model))
        efs = ExhaustiveFeatureSelector(model, scoring = metric, min_features = 7, max_features = 11, cv = cv)
        efs.fit(X_train, y_train)
        variables = X_train.columns[list(efs.best_idx_)]
        high_score = efs.best_score_
        t2 = time.time() - t1

    all_scores_dict = {"Selector" : a[-1], "Estimateur" : b[0], "Score" : high_score, "N_Variables": len(variables),"Variables" :list(variables), "Duree(s)" : t2 }    

    return all_scores_dict

def selection_list(selectors, algos, X_train, y_train, X_test, Forward = False, metric = "accuracy", 
                  smote = False, cv = 5):
    """
    Cette fonction affiche pour une liste de selectors et d'estimateurs, le nombre et les variables selectionnées de même que la métrique correspondant à chaque combinaison selector-estimator
    Si le selector est la RFE ou ExhaustiveFeatureSelector, la fonction affiche le score obtenu après la sélection.
    
    Paramètres :
    ------
    selectors : list of selectors
            Liste contenant des méthodes de sélection de variables,
                
    algos : list of estimators
            Liste contenant Tout algorithme de classification type sklearn

    X_train, y_train : DataFrame
            Training data

    X_test : DataFrame
            Variables explicatives contenues dans le test set

    Forward : bool (default: True)
            Forward selection si True,
    
    print_estimateur : bool (default: False)
            Affiche le nom de l'estimateur utilisé
    
    metric : str
            Métrique à optimiser

    smote : bool (default: False)
            Oversampling si True

    cv : int
            Nombre de KFold
    
    Returns
    ------
    recap : DataFrame
            Récapitulatifs des résultats 

    Examples
    ------
    >>>from sklearn.linear_model import LogisticRegression, LinearRegression
    >>>from sklearn.feature_selection import RFE, ExhaustiveFeatureSelector
    >>>selectors = [RFE, ExhaustiveFeatureSelector]
    >>>algos = [LogisticRegression, LinearRegression]
    selection_list(selectors, algos, print_estimateur = True)

    """
    dictionnaire = {"Selector" : [], "Estimateur" : [], "Score" : [], "N_Variables": [], "Duree(s)" : [], "Variables" : []}

    for algo in algos:
        for selector in selectors :
            all_scores_dict = selection_np(selector, algo, X_train, y_train, X_test,  Forward = True, metric=metric, smote = smote)
            dictionnaire["Selector"].append(all_scores_dict["Selector"])
            dictionnaire["Estimateur"].append(all_scores_dict["Estimateur"])
            dictionnaire["Score"].append(all_scores_dict["Score"])
            dictionnaire["N_Variables"].append(all_scores_dict["N_Variables"])
            dictionnaire["Variables"].append(all_scores_dict["Variables"])
            dictionnaire["Duree(s)"].append(all_scores_dict["Duree(s)"])
            
    recap  = pd.DataFrame(dictionnaire)
    recap.sort_values(by=["Score"], inplace=True, ascending=False)
    
    return recap

def val(X, y, smote = False):
    """
    Parametres
    ------

    Returns
    ------

    Example
    ------
    
    """
    if smote == True : 
        title = 'Logit Regression with smote results'
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=0 ,k_neighbors=15)
        X_s, y_s = smote.fit_resample(X, y)
        X_sc = tools.add_constant(X_s, prepend=True, has_constant='skip')
        log_reg = sm.Logit(y_s, X_sc).fit().summary(title = title)
    else : 
        X_c = tools.add_constant(X, prepend=True, has_constant='skip')
        log_reg = sm.Logit(y, X_c).fit().summary2()

    print(log_reg)
    return 


def overfit(metrics, X, y, smote = False):
    """
    docstring
    """
    if smote == True:
        print("SMOTE")
        smote = SMOTE(random_state=0 ,k_neighbors=15)
        X, y = smote.fit_resample(X, y)

    reg = LogisticRegression(penalty='l2', C=5, solver='saga', max_iter= 100, random_state=0)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    
    scores_auc = cross_val_score(reg, X, y, scoring = 'roc_auc', cv = cv)

    scores = cross_validate(reg, X, y, scoring = metrics,
                            cv = cv, return_train_score=True)

    AUROC = np.mean(scores_auc)
    GINI = AUROC * 2 - 1

    print('Mean Auc: %.2f' %(AUROC))
    print('Gini: %.2f' % (GINI))

    sc = pd.DataFrame(scores)

    return sc

def model_perf(X_train, y_train, X_test, y_test, 
          smote = False, show_roc = False, show_conf_matrix = False, cut_off = 0.5, show_prc = True) : 

    if smote == True:
        
        print("SMOTE")
        print("--------")

        smote = SMOTE(random_state=42 , k_neighbors=15)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    lr = LogisticRegression(penalty='l2', C=5, max_iter= 100, random_state=42).fit(X_train, y_train)
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
    print('Gini test: %.2f' % (Gini))
    print("--------")
    Gini_ = roc_auc_ * 2 - 1
    print('Gini train: %.2f' % (Gini_))
    print("--------")
    if show_roc == True:
        print('roc curve')
        
        plt.figure(figsize=(6,6))
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.plot(fpr, tpr, lw=3, label='roc curve - Test (auc = {:0.2f})'.format(roc_auc))
        plt.plot(fpr_, tpr_, lw=3, label='roc curve - Train (auc = {:0.2f})'.format(roc_auc_))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        #plt.title('roc curve ', fontsize=16)
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
        plt.plot(precision_, recall_, label='prc - Train')
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
    print('Meilleur cut-off test : %.2f' % (seuil_opt))
    print('Meilleur cut-off train : %.2f' % (seuil_opt_))
    
    return train_pred, y_test, test_pred, fpr, tpr, roc_auc

def scoring(X_train, y_train, X_test, y_test, classes, train, test, X,  base = 1000, pdo = 30):
    
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


def grid_search(model, X_train, y_train):

    if model == "RandomForestClassifier":
        estimateur =RandomForestClassifier()
        print("RandomForestClassifier")
        print("------")
        params = {"n_estimators":list(range(5, 26, 10)), 
                  'max_depth':list(range(25, 101, 25)), 
                "max_features" : list(range(3, X_train.shape[1]+1, 1))}
        gs = GridSearchCV(estimateur,
              param_grid=params,
              scoring="roc_auc", return_train_score=True, cv = 5)
        t1 = time.time()
        gs.fit(X_train, y_train)
        results = gs.cv_results_
        t2 = time.time() - t1
        
    if model == "KNeighborsClassifier":
        estimateur = KNeighborsClassifier()
        print("KNeighborsClassifier")
        print("------")
        params = {"n_neighbors":list(range(5, 50, 15))}
        gs = GridSearchCV(estimateur,
              param_grid=params,
              scoring="roc_auc", return_train_score=True, cv = 5)
        t1 = time.time()
        gs.fit(X_train, y_train)
        results = gs.cv_results_
        t2 = time.time() - t1
        
    if model == "DecisionTreeClassifier":
        estimateur = DecisionTreeClassifier()
        print("DecisionTreeClassifier")
        print("------")
        params = {'max_depth':list(range(2, 21, 2)), "max_features" : list(range(3, X_train.shape[1]+1, 1))}
        gs = GridSearchCV(estimateur,
              param_grid=params,
              scoring="roc_auc", return_train_score=True, cv = 5)
        t1 = time.time()
        gs.fit(X_train, y_train)
        results = gs.cv_results_
        t2 = time.time() - t1
    if model == "GradientBoostingClassifier":
        estimateur = GradientBoostingClassifier()
        print("GradientBoostingClassifier")
        print("------")
        params = {'n_estimators': list(range(5, 26, 10)),'learning_rate': [.05, .1, .2],'max_depth': [1, 3, 5]}
        gs = GridSearchCV(estimateur,
              param_grid=params,
              scoring="roc_auc", return_train_score=True, cv = 5)
        t1 = time.time()
        gs.fit(X_train, y_train)
        results = gs.cv_results_
        t2 = time.time() - t1
    if model == "LinearDiscriminantAnalysis":
        estimateur = LinearDiscriminantAnalysis()
        print("LinearDiscriminantAnalysis")
        print("------")
        params = {}
        gs = GridSearchCV(estimateur,
              param_grid=params,
              scoring="roc_auc", return_train_score=True, cv = 5)

        t1 = time.time()
        gs.fit(X_train, y_train)
        results = gs.cv_results_
        t2 = time.time() - t1
    print("L'opération a pris %.2fs" %t2)
    print("Les meilleurs paramètres sont:", gs.best_params_)
    print("Le meilleur score AUC est: %.2f" %gs.best_score_)
    
    return 
