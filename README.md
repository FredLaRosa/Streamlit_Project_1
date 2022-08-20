# Welcome to Streamlit_Project_1 :wave:

**Data application to predict the nature of banknotes.**

## How it work?

Program for detecting counterfeit banknotes using the geometric data given in a csv file.
    
- If an "id" variable is present, its values will be used in index. 
- The necessary variables to make the prediciton are: ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]. 
- The data to be predicted will undergo the same pre-processing -using the RobustScaler method from Sklearn- as those used to fit our prediction model.
- For the prediction, it's the logistic regression model "logit_full_rbs" -created with Sklearn- which will be used. 
- A dataframe will be returned as a result. It will include: geometric values; the nature of the banknotes (*True* or *False*) and the probabilities of prediction indicating whether the ticket is *True* and *False*.

The tested banknotes will display in a biplot, showing the rows coordinates (banknotes) and the circle of correlations obtained during the factor analysis of the dataset used to fit our prediction model.

## Try it :test_tube:

You can download the dataset [billets_production.csv](https://github.com/FredLaRosa/Streamlit_Project_1/blob/main/billets_production.csv) and try it on the data app.

You can access app data directly by clicking on the badge below.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/FredLaRosa/Streamlit_Project_1/main/Streamlit_app.py)

## Installation

To run this app you need to install the library [Streamlit](https://github.com/streamlit/streamlit).

Streamlit may be installed using pip...

```bash
pip install streamlit
streamlit hello
```

or conda.

```bash
conda install -c conda-forge streamlit
```

Streamlit can also be installed in a virtual environment on [Windows](https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment#on-windows), [Mac](https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment#on-mac--linux), and [Linux](https://github.com/streamlit/streamlit/wiki/Installing-in-a-virtual-environment#on-mac--linux).

## Running the app

To run this data application, you can run this command:
```bash
streamlit run https://github.com/FredLaRosa/Streamlit_Project_1/blob/main/Streamlit_app.py
```

## Documentation
The docstring is written in english.
This data app is written in french.
