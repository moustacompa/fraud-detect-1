import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

model = joblib.load("fraud_model_paysim.pkl")
st.set_page_config(page_title="Détection de Fraude - RandomForest Model", page_icon="💸", layout="centered")
st.markdown("""
    <style>
        .stMainBlockContainer {
            max-width: 90% !important;
            padding-left: 0rem;
            padding-right: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💸 Détection de transactions suspectes")
st.markdown("""
Cette interface permet de tester le modèle **RandomForestClassifier** amélioré par une optimisation via Optuna et déployé via Flask/FastAPI.
Remplissez les champs ci-dessous puis cliquez sur **Analyser la transaction**.
""")
col1, col2 = st.columns([0.4, 0.6])
with col1:

    # === Formulaire utilisateur ===
    type_tx = st.selectbox("Type de transaction", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    oldbalanceOrg = st.number_input("Ancien solde expéditeur", min_value=0.0)
    amount = st.number_input("Montant", min_value=0.0)
    newbalanceOrig = st.number_input("Nouveau solde expéditeur", min_value=0.0)
    oldbalanceDest = st.number_input("Ancien solde destinataire", min_value=0.0)
    newbalanceDest = st.number_input("Nouveau solde destinataire", min_value=0.0)
    if st.button("Analyser"):
        ok = True
        if not type_tx:
            st.error("Veuillew choisir le type de transaction")
            ok = False
        if not oldbalanceOrg:
            st.error("Veuillew renseigner l'ancien solde de l'expéditeur")
            ok = False
        if not newbalanceOrig:
            st.error("Veuillew renseigner le nouveau solde de l'expéditeur")
            ok = False
        if not oldbalanceDest:
            st.error("Veuillew renseigner l'ancien solde du destinataire")
            ok = False
        if not newbalanceDest:
            st.error("Veuillew renseigner le nouveau solde du destinataire")
            ok = False
        if not amount: 
            st.error("Veuillew le montant de la transaction")
            ok = False
        if (ok==True):
            data = {}
            #st.write(data)
            data["step"] = 1
            data["type"] = type_tx
            data["amount"] = amount
            data["oldbalanceOrg"] = oldbalanceOrg
            data["newbalanceOrig"] = newbalanceOrig
            data["oldbalanceDest"] = oldbalanceDest
            data["newbalanceDest"] = newbalanceDest
            data["isFlaggedFraud"] = 0
            data["type_CASH_OUT"] = data["type"]=="CASH_OUT"
            data["type_DEBIT"] = data["type"]=="DEBIT"
            data["type_PAYMENT"] = data["type"]=="PAYMENT"
            data["type_TRANSFER"] = data["type"]=="TRANSFER"
            #data["type_CASH_IN"] = data["type"]=="CASH_IN"
            data['diffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
            data['diffDest'] = data['newbalanceDest'] - data['oldbalanceDest']
            data['orig_to_amount'] = data['oldbalanceOrg'] / (data['amount'] + 1)
            data['dest_to_amount'] = data['oldbalanceDest'] / (data['amount'] + 1)
            data['same_account'] = 1
            
            del data['type']
        
            df = pd.DataFrame([data])
            prediction = model.predict(df)
            if (int(prediction[0])==0):
                st.markdown("✅ Transaction normale")
            else:
                st.markdown("⚠️ Transaction suspecte")
        else: 
            st.markdown("*Veuillez corriger les erreurs signalées et réessayer svp.")
    
with col2:
    #Chargement des données du dataset uniquement au 1er affichage du formulaire

    #if type_tx == "PAYMENT" and oldbalanceOrg and newbalanceOrig and oldbalanceDest and amount:
    
    if "loaded" not in st.session_state:
        st.session_state.loaded = True
        df = pd.read_csv("1_part_1_paysim_2017.csv")    
        df = df.drop('step', axis=1)
        df = df.drop('nameOrig', axis=1)
        df = df.drop('nameDest', axis=1)
        df = df.drop('isFlaggedFraud', axis=1)
        df0 = df[df["isFraud"] == 0].sample(n=5)
        df1 = df[df["isFraud"] == 1].sample(n=5)
        tmp = pd.concat([df0, df1], ignore_index=True).sample(frac=1)    
        st.session_state.tmp = tmp
    else :
        tmp = st.session_state.tmp
    # Affichage d'un extrait du dataset
        #tmp = tmp.reset_index(drop=True)
    st.dataframe(tmp.head(10))