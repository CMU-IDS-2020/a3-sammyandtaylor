import streamlit as st
import math
import pandas as pd
import numpy as np
from numpy import cov
#import os
import vega
#from vega_datasets import data
import altair as alt
import altair_viewer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from scipy import stats
from scipy.stats import boxcox
from scipy.stats import pearsonr 
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller




st.title('An Analysis of Opioid deaths in the U.S.')
df = pd.read_csv("Wide_Master.csv")
if st.checkbox('Display Opioid Data'):
    st.write(df.head(10))

df = pd.read_csv("opioid.csv")
alt.data_transformers.enable('json')

if df is not None:
    df['TotalCount'] = df['TotalCount'].fillna((df['TotalCount'].median()))
    newdf = df['TotalCount']
    newdf.index = df.Year
    newdf.head()
    training = newdf[newdf.index <= 2014]
    test = newdf[newdf.index >= 2014]
    training.index = pd.to_datetime(training.index, format='%Y')
    training = training.resample('Y').count()
    test.index = pd.to_datetime(test.index, format='%Y')
    test = test.resample('Y').count()
    model = ARIMA(training, order = (0,1,1))
    results = model.fit()
    resultplot = results.plot_predict(1,50)
    plt.xlabel('Year')
    plt.ylabel('Total Opioid Deaths')

choose_model = st.sidebar.selectbox("Choose Machine Learning Model or EDA", 
['Show Exploratory Data Analysis', 'Show Forecasting Results', 'Show Regression Analysis'])

if (choose_model == "Show Forecasting Results"):
    st.markdown("## **Forecasting Opioid Deaths through 2050**")
    st.markdown("### Predicted total deaths from Methadone, Heroine, and Opioids")
    if not st.checkbox('Hide Graph', False, key=1):
        st.write(resultplot)
    if st.checkbox('Review the Data, then check the box if you would like to start your own forecast. Put in the amount of years you would like to forecast into the future'):
        users_input = st.text_input('Enter a number between 2 and 200: ')
        try: 
            newplot = results.plot_predict(1, int(users_input))
            plt.xlabel('Year')
            plt.ylabel('Total Opioid Deaths')
            st.write(newplot)
        except: 
            st.markdown("### Please enter a number between 2 and 200")


df = pd.read_csv("Wide_Master.csv")

if (choose_model == "Show Regression Analysis"):
    st.markdown("## *Regression Analysis of Predictor Variables vs. Target*")
    st.markdown("### Predictor Variables in this case are: Income, Bachelor_Degree, Grad_Degree, HS_Grad, Less_Than_HS, Associates_Degree")
    options = st.selectbox('What chart would you like to see?', ('Total', 'Heroin', 'Methadone', 'Other Opioids'))
    keepvar = ['Income', 'Unemployment', 'Non_US_Born', 'Bachelor_Degree', 'Grad_Degree', 'HS_Grad', 'Less_Than_HS', 'Associates_Degree', 'Heroin', 'Other', 'Methadone', 'Total']
    newdf = df[keepvar]
    newdf = pd.melt(newdf, id_vars=['Income', 'Unemployment', 'Non_US_Born', 'Total','Heroin', 'Other', 'Methadone'], value_vars=['Bachelor_Degree', 'Grad_Degree', 'HS_Grad', 'Less_Than_HS', 'Associates_Degree'], var_name= 'Education', value_name='EducationCount').reset_index()
    if (options == "Total"):
        scales = alt.selection_interval(bind='scales')
        Scatter_Plot_Altair = alt.Chart(newdf).mark_point().encode(
                            x=alt.X('Total'), y=alt.Y('EducationCount', 
                            scale = alt.Scale(zero=False, padding=1)), 
                            color='Education', size = 'Income',
                            tooltip=('Income:N','EducationCount:N','EducationType:Q')
                            ).properties(width=850,height=600)
        reg = Scatter_Plot_Altair.transform_regression('Total', 'EducationCount', groupby=['Education']).mark_line()
        (Scatter_Plot_Altair + reg).add_selection(scales)
        st.altair_chart((Scatter_Plot_Altair + reg).add_selection(scales))
    if (options == "Heroin"):
        scales = alt.selection_interval(bind='scales')
        Scatter_Plot_Altair = alt.Chart(newdf).mark_point().encode(
                            x=alt.X('Heroin'), y=alt.Y('EducationCount', 
                            scale = alt.Scale(zero=False, padding=1)), 
                            color='Education', size = 'Income',
                            tooltip=('Income:N','EducationCount:N','EducationType:Q')
                            ).properties(width=850,height=600)
        reg = Scatter_Plot_Altair.transform_regression('Heroin', 'EducationCount', groupby=['Education']).mark_line()
        (Scatter_Plot_Altair + reg).add_selection(scales)
        st.altair_chart((Scatter_Plot_Altair + reg).add_selection(scales))
    if (options == "Methadone"):
        scales = alt.selection_interval(bind='scales')
        Scatter_Plot_Altair = alt.Chart(newdf).mark_point().encode(
                            x=alt.X('Methadone'), y=alt.Y('EducationCount', 
                            scale = alt.Scale(zero=False, padding=1)), 
                            color='Education', size = 'Income',
                            tooltip=('Income:N','EducationCount:N','EducationType:Q')
                            ).properties(width=850,height=600)
        reg = Scatter_Plot_Altair.transform_regression('Methadone', 'EducationCount', groupby=['Education']).mark_line()
        (Scatter_Plot_Altair + reg).add_selection(scales)
        st.altair_chart((Scatter_Plot_Altair + reg).add_selection(scales))
    if (options == "Other Opioids"):
        scales = alt.selection_interval(bind='scales')
        Scatter_Plot_Altair = alt.Chart(newdf).mark_point().encode(
                            x=alt.X('Other'), y=alt.Y('EducationCount', 
                            scale = alt.Scale(zero=False, padding=1)), 
                            color='Education', size = 'Income',
                            tooltip=('Income:N','EducationCount:N','EducationType:Q')
                            ).properties(width=850,height=600)
        reg = Scatter_Plot_Altair.transform_regression('Other', 'EducationCount', groupby=['Education']).mark_line()
        (Scatter_Plot_Altair + reg).add_selection(scales)
        st.altair_chart((Scatter_Plot_Altair + reg).add_selection(scales))

if (choose_model=='Show Exploratory Data Analysis'):
    df_w = pd.read_csv("Wide_Master.csv")
    states_list = (df_w.State.unique())
    #states_list.insert(0,None)
    #alt.data_transformers.disable_max_rows()

    dff = pd.read_csv('population_engineers_hurricanes.csv')
    state_id = []

    for x in dff['id']:
        state_id.append(x)
    for i,j in zip(state_id,df_w['State'].unique()):
        df_w.loc[df_w['State'] == j, 'id'] = i
    df_w['id'] = df_w['id'].astype(int)
    df_w1 = df_w[['State','Total','id','Year']]

    df_df = df_w1.groupby('State').Total.sum()
    df_df = df_df.reset_index()

    df_df1 = df_w1.groupby('State').id.mean()

    df_df1 = df_df1.reset_index()
    df_df1 = df_df1.drop(['State'],axis=1)
    df = pd.concat([df_df,df_df1],axis=1)
    
    checked = st.sidebar.checkbox("Show table data")
    if checked:
        st.write(df_w)

    year_slider = st.sidebar.slider('Year',2011,2017,step=1)


    brush = alt.selection_interval()
    multi_state = alt.selection_multi(fields=['State'])

    st.markdown(
        """<style>
            .chart {text-align: left !important}
        </style>
        """, unsafe_allow_html=True) 

    chart = alt.Chart(df_w).mark_circle(size=200
    ).transform_filter(
        alt.datum['Year'] == year_slider
    ).encode(
        x=alt.X('mean_x:Q', title='Average Income ($)',scale=alt.Scale(zero=False),axis=alt.Axis(titleFontSize=20,labelFontSize=14)),
        y=alt.Y('tot_y:Q', title='Number of Deaths from Opioid Overdose', scale=alt.Scale(zero=False),axis=alt.Axis(titleFontSize=20,labelFontSize=14)),
        color= alt.Color('State:N', legend=None),

        tooltip=[alt.Tooltip('State:N'),alt.Tooltip('tot_y:Q',title="Overdose Deaths"),alt.Tooltip('mean_x:Q',title="Income")]
    ).properties(
        width=1000,
        height=500
    ).transform_aggregate(
        tot_y='sum(Total)',
        mean_x='mean(Income)',
        groupby=['State']
    ).interactive().transform_filter(brush
    )
    st.write("### *Pan and zoom to see data points more clearly on the scatter plot*")
    #
    #
    #BAR
    bar = alt.Chart(df_w).mark_bar(
    ).transform_filter(
        alt.datum['Year'] == year_slider
    ).encode(
        x=alt.X('State:N',sort='-y', scale=alt.Scale(zero=False),axis=alt.Axis(titleFontSize=20,labelFontSize=14)),
        y=alt.Y('mean(Unemployment):Q',title='Unemployment Rate (%)', scale=alt.Scale(zero=False),axis=alt.Axis(titleFontSize=20,labelFontSize=14)),
        color= alt.condition(brush,'State:N',alt.value('lightgray'), legend=None),
        tooltip=[alt.Tooltip('State:N'),alt.Tooltip('mean(Unemployment):Q',title="Unemployment")]
    ).properties(
        width=1000,
        height=500
    ).add_selection(
        brush
    )

    states_us = alt.topo_feature('https://vega.github.io/vega-datasets/data/us-10m.json', 'states')
    source = 'https://raw.githubusercontent.com/sammyhajomar/test/main/altair-dataset.csv'
    variables = ['State','Total','id']


    US_map = alt.Chart(states_us).mark_geoshape().encode(
        color=alt.condition(multi_state,'Total:Q',alt.value('lightgray'),title='Total Deaths'),
        tooltip=['State:N',alt.Tooltip('Total:Q',title='Total Deaths')]
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(source,'id',variables),
    ).project(
        type='albersUsa'
    ).add_selection(multi_state
    ).properties(
        width=1200,
        height=850
    )


    #TEXT
    text = alt.Chart(df_w).mark_text(
        align= 'left',
        baseline = 'middle',
        dx= 10
    ).encode(
        x='mean(Income):Q',
        y='sum(Total):Q',
        text='State'
    ).transform_filter(
        alt.datum['Year'] == year_slider
    )


    st.write("## Total Opioid Overdose Deaths by U.S. State")
    chart + text & bar
    st.write("### *Select a subset of the bar graph for an interaction with the scatter plot*")
    st.write('\n')
    st.write('\n')
    st.write('## Total Opioid Overdose Deaths 2011-2017  by U.S. State')
    st.write(US_map)
