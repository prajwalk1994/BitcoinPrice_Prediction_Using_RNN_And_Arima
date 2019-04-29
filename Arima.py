from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
import numpy as np

def parser(x):
    dateout = []
    for time in x:
        dateout.append(datetime.fromtimestamp(int(time)))
    return dateout


series = read_csv('CMPE-256-Large-Scale-Analytics-/data/bitstamp.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# print(series.head(10))
# Split the data into individual dataframes
price = series.iloc[:, [6]].fillna(method='ffill').head(1000).values
open = series.iloc[:, [0]].fillna(method='ffill').head(1000).values
high = series.iloc[:, [1]].fillna(method='ffill').head(1000).values
low = series.iloc[:, [2]].fillna(method='ffill').head(1000).values
close = series.iloc[:, [3]].fillna(method='ffill').head(1000).values

# Split the individual dataframes to train and test
size = int(len(price) * 0.66)
trainPrice, testPrice = price[0:size], price[size:len(price)]
trainOpen, testOpen = open[0:size], open[size: len(price)]
trainHigh, testHigh = high[0:size], high[size: len(price)]
trainLow, testLow = low[0:size], low[size: len(price)]
trainClose, testClose = close[0:size], close[size: len(price)]

# Converting the dataframes into lists
historyPrice = [x for x in trainPrice]
historyOpen = [x for x in trainOpen]
historyHigh = [x for x in trainHigh]
historyLow = [x for x in trainLow]
historyClose = [x for x in trainClose]

# Predicting the price of Bitcoin
for t in range(testLen):
    # Models for all the fields in the training dataset
    priceModel = ARIMA(historyPrice, order=(5, 1, 0)).fit(disp=0)
    openModel = ARIMA(historyOpen, order=(5, 1, 0)).fit(disp=0)
    highModel = ARIMA(historyHigh, order=(5, 1, 0)).fit(disp=0)
    lowModel = ARIMA(historyLow, order=(5, 1, 0)).fit(disp=0)
    closeModel = ARIMA(historyClose, order=(5, 1, 0)).fit(disp=0)
    # Predict the future value
    outputPrice = priceModel.forecast()
    outputOpen = openModel.forecast()
    outputHigh = highModel.forecast()
    outputLow = lowModel.forecast()
    outputClose = closeModel.forecast()
    # Create a new list for all the predicted values
    predict = list()
    # Add the predicted values to the predicted list
    predict.append(outputPrice[0])
    predict.append(outputOpen[0])
    predict.append(outputHigh[0])
    predict.append(outputLow[0])
    predict.append(outputClose[0])
    # Append the output of the test to output object
    outputTest = list()
    outputTest.append(testPrice[t])
    outputTest.append(testOpen[t])
    outputTest.append(testHigh[t])
    outputTest.append(testLow[t])
    outputTest.append(testClose[t])
    # Add the test value to history object
    historyPrice.append(testPrice[t])
    historyOpen.append(testOpen[t])
    historyHigh.append(testHigh[t])
    historyLow.append(testLow[t])
    historyClose.append(testClose[t])
# Save the output to a csv file
np.savetxt("arimaOutput.csv", outputTest, delimiter=",", fmt='%s', header=header)
