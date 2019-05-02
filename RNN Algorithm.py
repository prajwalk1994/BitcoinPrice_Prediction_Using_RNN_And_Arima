#import python libraries
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Reading test and train data into dataframes
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train['date'] = pd.to_datetime(df_train['Timestamp'],unit='s').dt.date
df_test['date'] = pd.to_datetime(df_train['Timestamp'], unit = 's').dt.date

#Grouping data by date
group_train = df_train.groupby('date')
group_test = df_test.groupby('date')

#Finding mean price for both train and test data on each date
Real_Price_train = group_train['Weighted_Price'].mean()
Real_Price_test = group_test['Weighted_Price'].mean()

timestamps = group_test['date'].unique().values

#Data preprocess
df_train= Real_Price_train[:len(Real_Price_train)]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
A_train = training_set[0:len(training_set)-1]
b_train = training_set[1:len(training_set)]
A_train = np.reshape(A_train, (len(A_train), 1, 1))

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

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()
