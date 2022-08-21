
################################################################
# Importing functions and/or external classes:                 #
# ------------------------------------------------------------ #
#                                                              #
import streamlit as st
import pandas as pd
from PIL import Image
from prince import PCA as prince_PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.text import Text
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
# Setting the default "darkgrid" style
sns.set_style("darkgrid")
#                                                              #
################################################################

################################################################################
# Setting the style of the application on the web :                            #
# ---------------------------------------------------------------------------- #
#                                                                              #
st.set_page_config(page_title="Détecter les faux billets", page_icon=":euro:")
#                                                                              #
################################################################################

########################################################################################################
# Preparing the dataset to create the model:                                                           #
# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
banknotes_final = pd.read_csv(
    r"billets_final.csv")
                                                                                                      
rbs = RobustScaler()

banknotes_rbs = pd.DataFrame(rbs.fit_transform(banknotes_final.iloc[:,1:7]),
                           index=banknotes_final.index,
                           columns=banknotes_final.iloc[:,1:7].columns)

# Dependent variable
y_rbs = banknotes_final["is_genuine"]

# Quantitative independent variables
X_rbs = banknotes_rbs[[
    "diagonal", "height_left", "height_right", "margin_low", "margin_up",
    "length"
]]

# Fiting our model
logit = LogisticRegression(solver="newton-cg")
logit_full_rbs = logit.fit(X_rbs, y_rbs)
########################################################################################################

########################################################################################################
# Creating classes for our legend manager:                                                             #
# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
# For integers
class IntHandler:

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = Text(x0, y0, str(orig_handle), color="red", fontsize=16)
        handlebox.add_artist(text)
        return text


# For the ellipses
class HandlerEllipse(HandlerPatch):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center,
                             width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
#                                                                                                      #
########################################################################################################

########################################################################################################
# Data preparation for PCA:                                                                            #
# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
banknotes_PCA = banknotes_final.copy()
banknotes_PCA = banknotes_PCA.set_index("is_genuine")

std_not_scaled = StandardScaler(with_std=False)

banknotes_centered_not_scaled = pd.DataFrame(
    std_not_scaled.fit_transform(banknotes_PCA),
    index=banknotes_PCA.index,
    columns=banknotes_PCA.columns)

# Fiting the PCA
prince_pca = prince_PCA(n_components=2,
                        n_iter=3,
                        rescale_with_mean=True,
                        rescale_with_std=False,
                        copy=True,
                        check_input=True,
                        engine="auto",
                        random_state=42)
prince_pca = prince_pca.fit(banknotes_centered_not_scaled)
#                                                                                                      #
########################################################################################################

########################################################################################################
# Preparing data for biplot display:                                                                  #
# ---------------------------------------------------------------------------------------------------- #
#                                                                                                      #
# We label the name of the variables with a number
n_labels = [
    value
    for value in range(1, (len(banknotes_centered_not_scaled.index) + 1))
]

# Component coordinates
pcs = prince_pca.column_correlations(banknotes_centered_not_scaled)

# Row coordinates
pca_row_coord = prince_pca.row_coordinates(
    banknotes_centered_not_scaled).to_numpy()

# Preparing the colors for parameter "c"
colors = banknotes_centered_not_scaled.index.map({
    "True": 1,
    "False": 0
}).to_numpy()
#                                                                                                      #
########################################################################################################

########################################################################################################
# Building your application structure:                                                                 #
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
# Container configuration:                                                                             #
# -----------------------------------------------------------------------------------------------------#
#                                                                                                      #
with image:
    image = Image.open(
        r"Logo ONCFM.png"
    )
    st.image(image, caption="Logo ONCFM")

with header:
    st.title("Détection des faux billets")

with intro:
    st.subheader("Contexte")
    st.write(
        """Selon l'office central de la répression du faux monnayage (OCRFM), plus de 600 000 
faux billets de banque circulent en France chaque année. Avec les réseaux de la Camorra, du 
grand banditisme, des « officines » et du Darknet… il est assez facile aujourd'hui de se 
procurer de faux billets, se revendant en moyenne 70% moins cher de leur valeur d'origine. 

Sur l'ensemble des faux billets imprimés, 2/3 correspondent à des petites coupures de 20€ et 50€, 
générant des pertes financières pour les petits commerces, particuliers, ... impactés le plus
par ces escroqueries.

Pour accompagner les consommateurs, la Banque centrale européenne préconise d’appliquer une 
méthode (TRI) d’identification simple : la méthode Toucher, Regarder, Incliner. 
Le travail et le talent de certains faussaires, couplés aux progrès technologiques, ont rendu 
difficile la détection des faux billets grâce à l'œil nu et au toucher. 
Afin d'aider les personnes ayant un doute, des outils technologiques ont été créés afin de mieux les détecter :
- stylo à encre volatile, restant transparente et s’estompant sur un billet authentique, et se colorant si elle est appliquée sur un faux billet ;
- détecteurs de faux billets à lampe UV ou infrarouge, identifiant des composants anormaux sur des billets falsifiés ;
- détecteurs automatiques, capables d’analyser tous les points d’authenticité d'un billet.
""")

with presentation:
    st.subheader("Détection automatique grâce à la Data Science")
    st.write(
        """A l'instar de l'ingéniosité grandissante de certains faussaires, la Data Science évolue 
aussi, offrant un panel d'outils qui améliore la qualité de détection des faux billets, et ce, 
s'appuyant sur divers critères comme les dimensions géométriques de ces derniers.

En analysant un jeu de données comportant les données géométriques de 1500 billets, dont 1000 Vrai
et 500 Faux, nous avons pu -grâce à l'analyse factorielle et à des algorithmes de Machine Learning-
créer un modèle prédisant à 99,20% (score de performance moyen obtenu sur l'ensemble des tests réalisés,
oscillant entre 98.67% et 100%) la nature de billets de banques.

Nous avons créé une fonction permettant de retourner, pour tout fichier *.csv* (comme le fichier
**billets_production.csv** téléchargeable sur le **GitHub** du projet [ici](https://github.com/FredLaRosa/Streamlit_Project_1/blob/main/billets_production.csv)) contenant les données géométriques de billets de banques
et respectant une certaine nomenclature, un tableau indiquant pour chaque billet, ses dimensions,
sa nature -*True* pour vrai, *False* pour faux- ainsi que le score de probabilité attestant de sa nature.

Nous afficherons les billets testés dans un bibplot, ce dernier affichant la projection des individus
(billets) et du cercle des corrélations obtenu lors de l'analyse factorielle du jeu de donnée ayant servi
à ajuster notre modèle de prédiction.

L'ACP nous montre bien que la nature des billets est observable sur l'axe F1 avec les variables **length**
et **margin_low** qui en sont les mieux représentées dans le cercle des corrélations. Les billets ayant 
une petite longueur (**length**), une marge inférieure et supérieure (**margin_low** et **margin_up**), 
ainsi qu'une hauteur gauche-droite (**height_left** et **height_right**) plus grande, sont considérés
comme faux."""
    )

with loading:
    st.header("Utilisation du programme de prédiction")
    st.write(
        """Pour détecter la nature des billets, merci de cliquer sur l'onglet **Browse files**
et d'insérer votre jeu de donnée en format *csv*.""")
    sel_col, disp_col = st.columns(2)

    uploaded_file = sel_col.file_uploader("Uploader un fichier")
    ###################################################################################
    # Application of the prediction on the uploaded file:                             #
    # ------------------------------------------------------------------------------- #
    #                                                                                 #
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        dataset = dataset.dropna()
        if dataset.columns.str.contains("id", case=False).any():
            dataset = dataset.set_index("id")

        dataset = dataset.loc[:, [
            "diagonal", "height_left", "height_right", "margin_low",
            "margin_up", "length"
        ]]
        # Data normalization with rbs object used to center model data logit_full_rbs
        dataset_rbs = pd.DataFrame(rbs.transform(dataset),
                                   index=dataset.index,
                                   columns=dataset.columns)

        # Application du modèle de régression logistique "logit_full_rbs"
        tadam = logit_full_rbs.predict(dataset_rbs)

        # Creation of a dataframe returning the results of the prediction
        result = dataset.copy()
        result["Nature"] = tadam
        result[["Proba Faux", "Proba Vrai"
                  ]] = logit_full_rbs.predict_proba(dataset_rbs).round(2)
        result["Nature"] = result["Nature"].map({True: "True", False: "False"})
        #                                                                                 #
        ###################################################################################

        ###################################################################################
        # Display biplot of rows and variables with the banknotes tested:                 #
        # ------------------------------------------------------------------------------- #
        #                                                                                 #
        #                                                                                 #
        # Coordinates of the banknotes to be tested from the "dataset" file:              #
        # ---------------------------------------
        #
        # Preprocessing with the same model used to get the 
        # "banknotes_centered_not_scaled" dataframe
        dataset_centered_not_scaled = pd.DataFrame(
            std_not_scaled.transform(dataset),
            index=dataset.index,
            columns=dataset.columns)
        #                                                                                 #
        #                                                                                 #
        # Biplot settings:                                                                #
        # ---------------------------------------                                         #
        #                                                                                 #
        # Display row coordinates
        ax = prince_pca.plot_row_coordinates(
            banknotes_centered_not_scaled,
            figsize=(12, 12),
            x_component=0,
            y_component=1,
            labels=None,
            color_labels=banknotes_centered_not_scaled.index,
            ellipse_outline=True,
            ellipse_fill=True,
            show_points=True)

        # Display row coordinates for "dataset"
        ax.scatter(x=prince_pca.row_coordinates(
            dataset_centered_not_scaled)[0],
                   y=prince_pca.row_coordinates(
                       dataset_centered_not_scaled)[1],
                   color="#ffe66d",
                   marker="^",
                   s=265)

        # Display the variables in the correlation circle
        plt.quiver(np.zeros(pcs.to_numpy().shape[0]),
                   np.zeros(pcs.to_numpy().shape[0]),
                   pcs[0],
                   pcs[1],
                   angles="xy",
                   scale_units="xy",
                   scale=1,
                   color="r",
                   width=0.003)

        # Display variables names
        for i, (x, y) in enumerate(zip(pcs[0], pcs[1])):
            plt.text(x,
                     y,
                     n_labels[i],
                     fontsize=26,
                     ha="center",
                     va="bottom",
                     color="red")

        # Display id of the banknotes from dataset
        for i, (x, y) in enumerate(
                zip(
                    prince_pca.row_coordinates(
                        dataset_centered_not_scaled)[0],
                    prince_pca.row_coordinates(
                        dataset_centered_not_scaled)[1])):
            plt.text(x,
                     y,
                     dataset_centered_not_scaled.index[i],
                     fontsize=26,
                     ha="center",
                     va="bottom",
                     color="#ffe66d")

        # Display the circle
        circle = plt.Circle((0, 0),
                            1,
                            facecolor="none",
                            edgecolor="#9fc377",
                            linewidth=2,
                            label="Cercle des corrélations")
        # We add our circle to the graph
        plt.gca().add_artist(circle)

        # Title
        plt.title("Biplot des individus et des variables", fontsize=22)

        # X and Y labels with explained inertia
        plt.xlabel("F{} ({}%)".format(
            1, round(100 * prince_pca.explained_inertia_[0], 1)),
                   fontsize=16)
        plt.ylabel("F{} ({}%)".format(
            2, round(100 * prince_pca.explained_inertia_[1], 1)),
                   fontsize=16)

        # Legend of variables
        legend_1 = plt.legend(n_labels,
                              pcs.index,
                              handler_map={int: IntHandler()},
                              bbox_to_anchor=(1, 1),
                              fontsize=16)

        # Legend of the scatter plot
        true_patch = mpatches.Patch(color="#ff7f0e", label="True")
        false_patch = mpatches.Patch(color="#0272a2", label="False")
        banknotes_tested = mpatches.Patch(color="#ffe66d", label="Billets testés")

        plt.legend(handles=[true_patch, false_patch, banknotes_tested, circle],
                   handler_map={
                       int: IntHandler(),
                       mpatches.Circle: HandlerEllipse()
                   },
                   bbox_to_anchor=(1, 0.75),
                   fontsize=16)

        # Display legend_1
        plt.gca().add_artist(legend_1)

        # Display of a grid to facilitate the reading of the coordinates
        plt.grid(visible=True)
        #                                                                                 #
        ###################################################################################

        st.subheader("Résultat de la prédiction")
        # Display of the "result" DataFrame
        st.dataframe(
            result.style.format(subset=[
                "diagonal", "height_left", "height_right", "margin_low",
                "margin_up", "length", "Proba Faux", "Proba Vrai"
            ],
                                  formatter="{:.2f}"))

        st.subheader(
            "Affichage des billets testés sur le biplot des individus et variables"
        )
        # Display biplot
        st.pyplot()
        # Disabling warning for calling st.pyplot() without any arguments
        st.set_option("deprecation.showPyplotGlobalUse", False)
