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
prediction_days = 365
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
