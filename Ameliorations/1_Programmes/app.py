#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 00:24:37 2022

@author: Abdoul_Aziz_Berrada
"""

import streamlit as st
import pandas as pd
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


st.set_option('deprecation.showPyplotGlobalUse', False)



#--------------------------------- Head
## ===> Header

st.markdown("<h1 style='text-align: center;'> LOAN CENTER </h1>", unsafe_allow_html=True)



st.image('https://raw.githubusercontent.com/aadmberrada/Linux_HMEQ/main/Ameliorations/2_Data/loans.jpeg')

st.markdown("""
            
Le but de ce projet est de réaliser une grille de score permettant d'aider à la décision.
Pour la réalisation de ce projet, nous nous sommes inspirés d'un projet fait par [Carl Lejerskar](https://github.com/Carl-Lejerskar). 
Son projet est directement trouvable [ici](https://github.com/Carl-Lejerskar/HMEQ) """
)

st.markdown(""" Un projet de:
*   **Abdoul Aziz Berrada**
*   **Amira Slimene**
*   **Yasmine Ouyahya** """)

st.markdown(""" ==> Pre requis :
*   **Python :** pandas, sklearn, scorecardpy, streamlit.
*   **Source des données :** https://github.com/Carl-Lejerskar/HMEQ/blob/master/hmeq.csv""")



st.markdown("<h2 style='text-align: center;'> Formulaire de demande de prêt </h2>", unsafe_allow_html=True) 

PATH = "https://raw.githubusercontent.com/aadmberrada/Linux_HMEQ/main/Ameliorations/2_Data/"

data = pd.read_csv(PATH + "train_clean.csv").drop("Unnamed: 0", axis = 1)


LOAN = st.text_input("Entrez le montant du prêt demandé en $")

MORTDUE = st.text_input("Entrez le montant dû sur l’hypothèque existante en $")
st.write("Indication : La moyenne est de ", round(data.MORTDUE.mean()), "$")

VALUE = st.text_input("Entrez la valeur du bien actuel en $")
st.write("Indication : La moyenne est de ", round(data.VALUE.mean()), "$")

REASON = st.selectbox('Indiquez le but du prêt', list(data.REASON.unique()))
JOB = st.selectbox('Indiquez votre catégorie professionnelle', list(data.JOB.unique()))

YOJ = st.text_input("Entrez le nombre d’années dans l’emploi actuel")
#st.write("Indication : La moyenne est de ", int(data.YOJ.mean()))

DEROG = st.text_input("Entrez le nombre de cas dérogatoires majeurs")
st.write("Indication : La moyenne est de ", int(data.DEROG.mean()))

DELINQ = st.text_input("Entrez le nombre de lignes de crédit en défaut de paiement")
st.write("Indication : La moyenne est de ", int(data.DELINQ.mean()))

CLAGE = st.text_input("Entrez l'âge de la ligne de crédit la plus ancienne en mois")
st.write("Indication : La moyenne est de ", int(data.CLAGE.mean()), 'mois')

NINQ = st.text_input("Entrez le nombre d’enquêtes de crédit récentes")
st.write("Indication : La moyenne est de ", int(data.NINQ.mean()))

CLNO = st.text_input("Entrez le nombre de lignes de crédit")
st.write("Indication : La moyenne est de ", int(data.CLNO.mean()))

DEBTINC = st.text_input("Entrez votre ratio d'endettement en %")
st.write("Indication : La moyenne est de ", round(data.DEBTINC.mean(), 2))


inputs = {"LOAN":LOAN, "MORTDUE":MORTDUE, "VALUE":VALUE, "REASON":REASON, 
        "JOB":JOB,  "YOJ" :YOJ, "DEROG":DEROG, "DELINQ":DELINQ,
        "CLAGE":CLAGE, "NINQ":NINQ, "CLNO":CLNO, "DEBTINC":DEBTINC}


st.write("Veuillez confirmez l'exactitude des données entrées",  inputs)



def scoring(X, base = 1000, pdo = 30):
    
    #CREATION DE LA GRILLE SCORE À PARTIR DES CLASSES, DU MODÈLE M ET CALIBRAGE SUR "base" POINTS
    lr = LogisticRegression(penalty='l2', C=5, max_iter= 100, random_state=42)
    lr.fit(x, y)
    test_pred = lr.predict_proba(test_woe)[:,1]
    
    ppb = sc.scorecard(classes, lr, xcolumns=test_woe.columns, 
                       odds0=1/500,  points0=base, pdo=pdo, basepoints_eq0 = True)
    
    #CALCUL DES SCORES TOTAUX DANS LE TRAIN SET
    train_score = sc.scorecard_ply(train_woe, ppb, print_step=0, only_total_score=False)
    
    return test_pred


score = st.button("Appuyez pour voir votre crédit score")
if score:

    if (any(val for val in inputs.values())!="") == True:
        
        inputs_ = {"LOAN":int(LOAN), "MORTDUE": int(MORTDUE), "VALUE":int(VALUE), "REASON":REASON, 
        "JOB":JOB,  "YOJ" :int(YOJ), "DEROG":int(DEROG), "DELINQ":int(DELINQ),
        "CLAGE":int(CLAGE), "NINQ":int(NINQ), "CLNO":int(CLNO), "DEBTINC": round(DEBTINC, 2)}

        df = pd.DataFrame(data = inputs_, index=[0])
        st.table(df)
        classes = sc.woebin(data, y="BAD", positive=1, method="chimerge")
        train_woe = sc.woebin_ply(data, classes)
        test_woe = sc.woebin_ply(df, classes)
        tr = train_woe.drop("BAD", axis =1)
        test_woe = test_woe[tr.columns]
        x = train_woe.drop("BAD", axis = 1)
        y = train_woe["BAD"]       
        def scoring(base = 1000, pdo = 30):
            
            #CREATION DE LA GRILLE SCORE À PARTIR DES CLASSES, DU MODÈLE M ET CALIBRAGE SUR "base" POINTS
            lr = LogisticRegression(penalty='l2', C=5, max_iter= 100, random_state=42)
            lr.fit(x, y)
            test_pred = lr.predict_proba(test_woe)[:,1]
            
            ppb = sc.scorecard(classes, lr, xcolumns=test_woe.columns, 
                               odds0=1/500,  points0=base, pdo=pdo, basepoints_eq0 = True)
            
            #CALCUL DES SCORES TOTAUX DANS LE TRAIN SET
            train_score = sc.scorecard_ply(data, ppb, print_step=0, only_total_score=False)
            test_score = sc.scorecard_ply(df, ppb, print_step=0, only_total_score=False)
            
            return test_pred, test_score
        test_pred, test_score = scoring(base = 1000, pdo = 30)

        st.write("Votre score est de", int(test_score.loc[ 0, "score"]),"points.", " Ce nombre de points vous donne une probabilité de défaut de", round(100*test_pred[0], 2), "%")
        
    if (any(val for val in inputs.values())!="") == False:
        st.warning("Veuillez remplir tous les champs")
