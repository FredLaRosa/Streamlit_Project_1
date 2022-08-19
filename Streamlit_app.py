#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#####################################################
# Importation des fonctions et/ou classes externes: #
# ------------------------------------------------- #
#                                                   #
import streamlit as st                              #
import pandas as pd                                 #
import joblib                                       #
from PIL import Image                               #
#                                                   #
#####################################################


########################################################################################################
# Préparation du dataset pour créer le modèle:                                                         #
# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
billets_final = pd.read_csv(
    r"billets_final.csv")
                                                                                                      
# Importation du modele de prediction "logit_full_rbs" et de
# l'objet de preprocessing "rbs"
logit_full_rbs, rbs = joblib.load(
    r"logit_predict_nature_billets.joblib")
########################################################################################################


########################################################################################################
# Création des conteneurs:                                                                             #
# -----------------------------------------------------------------------------------------------------#
#                                                                                                      #
image = st.container()
header = st.container()
intro = st.container()
presentation = st.container()
loading = st.container()
#                                                                                                      #
########################################################################################################


########################################################################################################
# Configuration des conteneurs:                                                                        #
# -----------------------------------------------------------------------------------------------------#
#                                                                                                      #
with image:
    image = Image.open(
        r"Logo ONCFM.png")
    st.image(image, caption="Logo ONCFM")


with header:
    st.title("Détection des faux billets")

with intro:
    st.subheader("Contexte")
    st.text(
        """Selon l'office central de la répression du faux monnayage (OCRFM), plus de 600 000
faux billets de banque circulent en France chaque année. Avec les réseaux de la
Camorra, du grand banditisme, des «officines» et Darknet… il est assez facile
aujourd'hui de se procurer de faux billets, se revendant en moyenne 70% moins
cher de leur valeur d'origine. 

Sur l'ensemble des faux billets imprimés, 2/3 correspondent à des petites
coupures de 20€ et 50€, générant des pertes financières pour les petits commerces,
particuliers,... impactés le plus par ces escroqueries.

Pour accompagner les consommateurs, la Banque centrale européenne préconise
d’appliquer une méthode (TRI) d’identification simple: la méthode Toucher,
Regarder, Incliner. Le travail et le talent de certains faussaires, couplés aux 
progrès technologiques, ont rendu difficile la détection des faux billets grâce à
l'oeil nu et au toucher. Pour aider les personnes ayant un doute, des outils
technologiques ont été créés afin de mieux les détecter:
    - stylo à encre volatile, restant transparente et s’estompant sur un billet
    authentique, et se colorant si elle est appliquée sur un faux billet;
    - détecteurs de faux billets à lampe UV ou infrarouge, identifiant des 
composants anormaux sur des billets falsifiés;
    - détecteurs automatiques, capables d’analyser tous les points d’authenticité
d'un billet."""
    )

with presentation:
    st.subheader("Détection automatique grâce à la Data Science")
    st.text(
        """A l'instar de l'ingéniosité grandissante de certains faussaires, la
Data Science évolue aussi, offrant un pannel d'outils qui améliore la qualité de
détection des faux billets, et ce, s'appuyant sur divers critères comme les
dimensions géométriques de ces derniers.

En analysant un jeu de données comportant les données géométriques de 1500
billets, dont 1000 Vrai et 500 Faux, nous avons pu -grâce à l'analyse factorielle
et à des algorithmes de Machine Learning- créer un modèle prédisant à 
99,20% (score de performance moyen obtenu sur l'ensemble des tests réalisés,
oscillant entre 98.67% et 100%) la nature de billets de banques.

Nous avons créé un fonction permettant de retourner, pour tout fichier '.csv'
(comme le fichier 'billets_production.csv' présenté ci-dessous)contenant les
données géométriques de billets de banques et respectant une certaine nomenclature,
un tableau indiquant pour chaque billet, ses dimensions, sa nature -True pour vrai,
False pour faux- ainsi que le score de probabilité attestant de sa nature.""")

with loading:
    st.header("Utilisation du programme de prédiction")
    st.text(
        """Pour détecter la nature des billets, merci de cliquer sur l'onglet
"Browse files" et d'insérer votre jeu de donnée en format csv.""")
    sel_col, disp_col = st.columns(2)

    uploaded_file = sel_col.file_uploader("Uploader un fichier")
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        dataset = dataset.dropna()
        if dataset.columns.str.contains("id", case=False).any():
            dataset = dataset.set_index("id")

        dataset = dataset.loc[:, [
            "diagonal", "height_left", "height_right", "margin_low",
            "margin_up", "length"
        ]]
        # Normalisation des données avec l'objet rbs utilisé pour centrer
        # les données du modèle logit_full_rbs
        dataset_rbs = pd.DataFrame(rbs.transform(dataset),
                                   index=dataset.index,
                                   columns=dataset.columns)

        # Application du modèle de régression logistique "logit_full_rbs"
        tadam = logit_full_rbs.predict(dataset_rbs)

        # Création d'un df retournant les résultats de la prédiction
        resultat = dataset.copy()
        resultat["Nature"] = tadam
        resultat[["Proba Faux", "Proba Vrai"
                  ]] = logit_full_rbs.predict_proba(dataset_rbs).round(2)
        resultat["Nature"] = resultat["Nature"].map({
            True: "True",
            False: "False"
        })
        st.dataframe(
            resultat.style.format(subset=[
                "diagonal", "height_left", "height_right", "margin_low",
                "margin_up", "length", "Proba Faux", "Proba Vrai"
            ],
                                  formatter="{:.2f}"))
