import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt
import seaborn as sns
#from scipy.stats import shapiro
from scipy import stats
from matplotlib import pyplot
#from scipy.stats import boxcox
from numpy import cov
from scipy.stats import pearsonr 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import math


st.title('This is our new Project')
df = pd.read_csv("opioid.csv")
if st.checkbox('Display Opiod Data'):
    st.write(df.head(10))

#I. Exploratory Data Analysis
    #a. Observe the columns
#print(df.columns)
    #b. Determine the Categorical and Continuous Variables
        # Continuous: Income Unemployment NonUSBorn
        # Prediction for CountbyType 
idvariable = ['ID']
targetvariable = ['CountByType']
Continuousvar = ['Income', 'Unemployment', 'NonUSBorn']
allnum = ['Income', 'Unemployment', 'NonUSBorn', 'CountByType']
    #c. Univariate Analysis: 
#df.describe()
#df.info()
#sns.distplot(df.Income.dropna(), kde=False, bins = 50)
#sns.distplot(df.NonUSBorn.dropna(), kde=False, bins = 50)
#sns.distplot(df.Unemployment.dropna(), kde=False, bins = 50)
        #Distributions look pretty skewed. Lets check this out with stats
        #Normality tests
#normaldf = df[allnum]
#normaldf.head(100)
#normaldf.drop_duplicates(inplace=True)
#incomecol = normaldf['Income'].dropna()
#emplcol = normaldf['Unemployment'].dropna()
#nonuscol = normaldf['NonUSBorn'].dropna()
#targetcol = df['CountByType'].dropna()
#stat, p = shapiro(targetcol)
#print('Target Statistics=%e, p=%e' % (stat, p))
#stat, p = shapiro(incomecol)
#print('Income Statistics=%e, p=%e' % (stat, p))
#stat, p = shapiro(emplcol)
#print('Unemployment Statistics=%e, p=%e' % (stat, p))
#stat, p = shapiro(nonuscol)
#print('NonUSBorn Statistics=%e, p=%e' % (stat, p))
        # With an alpha of .05, none of the data is normally distributed
        #Transform the data with boxcox 
#df['Income'] = boxcox(df['Income'], 0)
#pyplot.hist(df['Income'])
#df['Unemployment'] = boxcox(df['Unemployment'], 0)
#pyplot.hist(df['Unemployment'])
#df['NonUSBorn'].min()
#xt, lmbda = stats.probplot(df['NonUSBorn'], dist=stats.norm)
#pyplot.hist(df['CountByType'])
#df['CountByType'] = boxcox(df['CountByType'], 0)
#pyplot.hist(df['CountByType'])
    #d. Bivariate Analysis:
        #Correlation Analysis: 
#sns.pairplot(df, y_vars=Continuousvar, x_vars=targetvariable)
            #Plots show possible positive correlation 
            #Needs further exploration
            #But first: Encode Categorical variable: County
        #Covariance Analysis: 
#rho, pvalue = stats.spearmanr(df['CountByType'], df['Unemployment'], nan_policy='omit')
#print(rho, pvalue)
#rho, pvalue = stats.spearmanr(df['CountByType'], df['Income'], nan_policy='omit')
#print(rho, pvalue)
#rho, pvalue = stats.spearmanr(df['CountByType'], df['NonUSBorn'], nan_policy='omit')
#print(rho, pvalue)
            #Looks like the only significant relationship is citizenship
        #Encode Categorical variable: County
#df['County'] = df['County'].astype('category')
#df['county_cat'] = df['County'].cat.codes
#model = ols('CountByType ~ County', data = df).fit()
#anova = sm.stats.anova_lm(model, typ=2)
#print(anova)
#II. Handle Missing Values
    #a. Handle Missing Values for CountByType
#df.isnull().sum()
#df[['Type', 'CountByType']].groupby('Type').median()
#Heroin_Med = 22.0
#Methadone_Med = 16.0 
#Other_Opioid_Med = 19.0
#print(len(df['CountByType']))
#criteria1 = (df.Type == 'Heroin')
#df.loc[df['CountByType'].isnull() & criteria1, 'CountbyType'] = Heroin_Med
#criteria2 = (df.Type == 'Methadone')
#df.loc[df['CountByType'].isnull() & criteria2, 'CountbyType'] = Methadone_Med
#criteria3 = (df.Type == 'Other Opiod')
#df.loc[df['CountByType'].isnull() & criteria3, 'CountbyType'] = Other_Opioid_Med
        #Missing values are replaced with category median 
    #b. Handle Missing Values for Income 
#df['Income'] = df['Income'].fillna((df['Income'].median()))
#df[Continuousvar].isnull().any()
    #c. Handle NonUSBorn Missing Values
#df['NonUSBorn'] = df['NonUSBorn'].fillna((df['NonUSBorn'].median()))
    #d. Handle Unemplyment Missing Values 
#df['Unemployment'] = df['Unemployment'].fillna((df['Unemployment'].median()))    
#III. Divide Data into Training and Validation Datasets
#X, y = df[Continuousvar], df[targetvariable]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
#IV. Linear Model
#Continuousvar = ['Income', 'Unemployment', 'NonUSBorn', 'county_cat']
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#regressor.score(X_test, y_test)
    #This model only has .13% predictive power
#V. Time Series Forecasting
df['TotalCount'] = df['TotalCount'].fillna((df['TotalCount'].median()))
#df['Year'].value_counts()
newdf = df['TotalCount']
newdf.index = df.Year
newdf.head()
training = newdf[newdf.index <= 2014]
test = newdf[newdf.index >= 2014]
training.index = pd.to_datetime(training.index, format='%Y')
training = training.resample('Y').count()
test.index = pd.to_datetime(test.index, format='%Y')
test = test.resample('Y').count()
    #First, a plot of the data 
training.plot(figsize=(20,14), title= 'Death Count by Year', fontsize=16)
test.plot(figsize=(20,14), title= 'Death Count by Year', fontsize=16)
    #Now some exponential smoothing 
#y_hat_avg = test.copy()
#model = SimpleExpSmoothing(np.asarray(training))
#fitting = model.fit(smoothing_level=0.6,optimized=False)
#y_hat_avg['SES'] = fitting.forecast(len(test))

#plt.figure(figsize = (16,8))
#plt.plot(training, label = 'Training')
#plt.plot(test, label='Test')
#plt.plot(y_hat_avg['SES'], label='SES')
#plt.legend(loc='best')
#plt.show()
    #Use Model for Forecasting
y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(training), trend='add', seasonal_periods=1).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(training, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
