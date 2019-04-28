import pandas as pd
df = pd.read_csv('C:\\Users\\Aishwariya\\Desktop\\CMPE256 Project\\test.csv')
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from matplotlib import pyplot
#from pandas.tools.plotting import autocorrelation_plot
df = pd.read_csv('C:\Users\Aishwariya\Desktop\CMPE256 Project\test.csv')
df

df.fillna(method='ffill', inplace=True)
df 
df


def parser(x):
	return pd.to_datetime(x, unit='s')

series = read_csv('C:\\Users\\Aishwariya\\Desktop\\CMPE256 Project\\test.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.fillna(method='ffill', inplace=True)
series
#autocorrelation_plot(series)
#pyplot.show()
series1 = series.iloc[:,[6]]
series1

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv")
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()