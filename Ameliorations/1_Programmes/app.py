
import streamlit as st
import pandas as pd
import time
import numpy as np
import scorecardpy as sc
from sklearn.model_selection import train_test_split


st.set_option('deprecation.showPyplotGlobalUse', False)



#--------------------------------- Head
## ===> Header
#st.title("")
st.markdown("<h1 style='text-align: center;'> LOAN CENTER </h1>", unsafe_allow_html=True)
#st.title("Stock Value Prediction using Neural Networks")


st.image('/Users/abdoul_aziz_berrada/Documents/M2_MOSEF/2_Projets/Semestre1/Linux_HMEQ/Ameliorations/2_Data/loans.jpeg')

st.markdown("""
            
Le but de ce projet est de réaliser une grille de score permettant d'aider à la décision.

Pour la réalisation de ce projet, nous nous sommes inspirés d'un projet fait par [Carl Lejerskar](https://github.com/Carl-Lejerskar). 
Son projet est directement trouvable [ici](https://github.com/Carl-Lejerskar/HMEQ) """
)


st.markdown(""" ==> Pre requis :
*   **Python :** pandas, sklearn, scorecardpy, streamlit.
*   **Source des données :** https://github.com/Carl-Lejerskar/HMEQ/blob/master/hmeq.csv""")



st.markdown("<h2 style='text-align: center;'> Formulaire de demande de prêt </h2>", unsafe_allow_html=True) 

PATH = "/Users/abdoul_aziz_berrada/Documents/M2_MOSEF/2_Projets/Semestre1/Linux_HMEQ/Ameliorations/2_Data/"
data = pd.read_csv(PATH + "data_clean.csv").drop("Unnamed: 0", axis = 1)


LOAN = st.text_input("Entrez le montant du prêt demandé en $")
MORTDUE = st.text_input("Entrez le montant dû sur l’hypothèque existante en $")
VALUE = st.text_input("Entrez la valeur du bien actuel en $")
REASON = st.selectbox('Indiquez le but du prêt', list(data.REASON.unique()))
JOB = st.selectbox('Indiquez votre catégorie professionnelle', list(data.JOB.unique()))
YOJ = st.text_input("Entrez le nombre d’années dans l’emploi actuel")
DEROG = st.text_input("Entrez le nombre de cas dérogatoires majeurs")
DELINQ = st.text_input("Entrez le nombre de lignes de crédit en défaut de paiement")
CLAGE = st.text_input("Entrez l'âge de la ligne de crédit la plus ancienne en mois")
NINQ = st.text_input("Entrez le nombre d’enquêtes de crédit récentes")
CLNO = st.text_input("Entrez le nombre de lignes de crédit")
DEBTINC = st.text_input("Entrez votre ratio d'endettement")


inputs = {"LOAN":LOAN, "MORTDUE":MORTDUE, "VALUE":VALUE, "REASON":REASON, 
        "JOB":JOB,  "YOJ" :YOJ, "DEROG":DEROG, "DELINQ":DELINQ,
        "CLAGE":CLAGE, "NINQ":NINQ, "CLNO":CLNO, "DEBTINC":DEBTINC}


st.write("Veuillez confirmez l'exactitude des données entrées",  inputs)

score = st.button("Appuyez pour voir votre crédit score")
if score:

    if (any(val for val in inputs.values())!="") == True:
        inputs_ = {"LOAN":int(LOAN), "MORTDUE": int(MORTDUE), "VALUE":int(VALUE), "REASON":REASON, 
        "JOB":JOB,  "YOJ" :int(YOJ), "DEROG":int(DEROG), "DELINQ":int(DELINQ),
        "CLAGE":int(CLAGE), "NINQ":int(NINQ), "CLNO":int(CLNO), "DEBTINC": float(DEBTINC)}

        df = pd.DataFrame(data = inputs_, index=[0])
        st.table(df)
        st.dataframe(data)

        train, test = train_test_split(data, test_size = 0.2, random_state = 42)
        classes = sc.woebin(data, y="BAD", positive=1, method="chimerge")
        train_woe = sc.woebin_ply(train, classes)
        test_woe = sc.woebin_ply(test, classes)

        #test_woe = sc.woebin_ply(df, classes)
        #st.write(data.shape)





    if (any(val for val in inputs.values())!="") == False:
        st.warning("Veuillez remplir tous les champs")

