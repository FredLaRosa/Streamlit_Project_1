# Welcome to Streamlit_Project_1 :wave:

**Data application to predict the nature of banknotes.**

## How it work?

Counterfeit banknote detection program based on its data geometry written in a csv file.
    
- If an "id" variable is present, its values will be used in index. 
- The necessary variables are: ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]. 
- Data will undergo a preprocessing the RobustScaler method used for the feature scaling data when creating our model.
- For the prediction, it is the logistic regression model "logit_full_rbs", created with sklearn, which will be used. 
- A dataframe will be returned as a result, it will include: geometric values; the nature of the ticket (True=True, False=False) and the probabilities of prediction indicating whether the ticket is *True* and *False*.

## Try it :test_tube:

You can download the dataset [billets_production.csv](https://github.com/FredLaRosa/Streamlit_Project_1/blob/main/billets_production.csv) and try it on the data app.

You can directly to the app data by clicking on the badge below.

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

This data app is written in french.
