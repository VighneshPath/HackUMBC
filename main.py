import streamlit as st

from pmdarima import auto_arima
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from sklearn.linear_model import LinearRegression


days_to_predict = 1

st.title("Prediction of Daily Covid-19 Cases")

days_to_predict = st.select_slider("No. of Days ahead to predict", range(1,8), value = 3)

st.sidebar.title("Choose a method to predict Daily Covid-19 Cases")
chart_visual = st.sidebar.selectbox('Select Algorithm', 
                                    ('Auto-Regressive Integrated Moving-Average', 'Linear Regression'))

new_cases = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/new_cases.csv"

cases_df = pd.read_csv(new_cases)

india_df = pd.DataFrame()
india_df["new_cases"] = cases_df["India"][:-1]

usa_df = pd.DataFrame()
usa_df["new_cases"] = cases_df["United States"][:-1]

india_df.index = pd.to_datetime(cases_df["date"][:-1])

usa_df.index = pd.to_datetime(cases_df["date"][:-1])

india_df.dropna(inplace = True)

usa_df.dropna(inplace = True)


def lr_predict(dfs, n): 
    """
        A Function which takes an List of dataframes, and an integer between 1 to 7 for number of days for prediction.
        This is limited because if the number of days increases, the model may take alot of time to compute.
        Returns an List of arrays, consisting of the "days_ahead" values predictions of the given dataframes.
    """
    start_date = dfs[0].iloc[-1].name
    predictions = []
    for df in dfs:
        next_date = len(df) + 1
        
        x = np.arange(len(df))
        y=df.values

        x=x.reshape(-1,1)

        regressor = LinearRegression()
        regressor.fit(x,y) 
        index_future_dates = pd.date_range(start = start_date + timedelta(days=1), end = start_date + timedelta(days=(n)))
        temp = []
        for i in range(next_date, next_date + n):
            temp.append(regressor.predict([[i]])[0][0])
        pred = pd.DataFrame(temp, index = index_future_dates)
        pred.index = index_future_dates.strftime('%d/%m/%Y')
        predictions.append(pred)
    return predictions



def arima_predict(dfs, days_ahead=3):
    """
        A Function which takes an List of dataframes, and an integer between 1 to 7 for number of days for prediction.
        This is limited because if the number of days increases, the model may take alot of time to compute.
        Returns an List of arrays, consisting of the "days_ahead" values predictions of the given dataframes.
    """
    
    start_date = dfs[0].iloc[-1].name
    
    predictions = []
    
    for df in dfs:
        # Taking into consideration only recent cases to avoid overfitting
        df = df[-150:]
        
        warnings.filterwarnings("ignore")


        stepwise_fit = auto_arima(df["new_cases"], stepwise=True, trace = True, suppress_warnings = True)
        
        summary_string = str(stepwise_fit.summary())
        
        param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)',summary_string)
        
        p,d,q = int(param[0][0]) , int(param[0][1]) , int(param[0][2])

        model=ARIMA(df['new_cases'],order=(p, d, q))
        model_fit=model.fit()

        model_fit.summary()
        
        pred = model_fit.predict(start = len(df), end = len(df) + days_ahead - 1, typ = 'levels')

        index_future_dates = pd.date_range(start = start_date + timedelta(days=1), end = start_date + timedelta(days=(days_ahead)))

        pred.index = index_future_dates 

        pred.index = index_future_dates.strftime('%d/%m/%Y')

        predictions.append([pred])
    return predictions

if(chart_visual == "Auto-Regressive Integrated Moving-Average"):
    final_predictions = arima_predict([india_df, usa_df], days_to_predict)

    st.markdown("Using ARIMA")

    countries = ["India", "United States"]

    for pred in range(len(final_predictions)):
        for i in range(len(final_predictions[pred])):
            
            st.markdown(f"## {countries[pred]}")

            st.markdown(f"### Values Predicted")

            df = pd.DataFrame({'Dates':final_predictions[pred][i].index, 'Predictions':final_predictions[pred][i].values})
            st.dataframe(df)

            st.markdown(f"### Visualisation")

            fig = plt.figure(figsize = (8, 2), dpi = 100) 
            ax = fig.add_axes([0,0,1,1])

            ax.plot(final_predictions[pred][i],  label = "ARIMA")
            ax.legend(loc='best')
            st.pyplot(fig)

elif(chart_visual == "Linear Regression"):
    final_predictionlr = lr_predict([india_df, usa_df], days_to_predict)

    st.markdown("Using Linear Regression")

    countries = ["India", "United States"]

    for i in range(len(final_predictionlr)):
            
        st.markdown(f"## {countries[i]}")

        st.markdown(f"### Values Predicted")

        df = pd.DataFrame({'Dates':final_predictionlr[i].index, 'Predictions':final_predictionlr[i][final_predictionlr[i].columns[0]].values})
        st.dataframe(df)

        st.markdown(f"### Visualisation")

        fig = plt.figure(figsize = (8, 2), dpi = 100) 
        ax = fig.add_axes([0,0,1,1])

        ax.plot(final_predictionlr[i],  label = "Linear Regression")
        ax.legend(loc='best')
        st.pyplot(fig)
