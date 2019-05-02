from pandas import read_csv
from pandas import datetime, DataFrame
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
import csv

def parser(x):
    dateout = []
    for time in x:
        dateout.append(datetime.fromtimestamp(int(time)))
    return dateout


series = read_csv('data/train.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# print(series.head(10))
# Split the data into individual dataframes
price = series.iloc[:, [6]].fillna(method='ffill')
open = series.iloc[:, [0]].fillna(method='ffill')
high = series.iloc[:, [1]].fillna(method='ffill')
low = series.iloc[:, [2]].fillna(method='ffill')
close = series.iloc[:, [3]].fillna(method='ffill')

# Get train values
size = int(len(price) * 0.66)
trainPrice = price.values
trainOpen = open.values
trainHigh = high.values
trainLow = low.values
trainClose = close.values

#Read test data from CSV file
testData = read_csv(
    'data/test.csv', parse_dates=[0], squeeze=True, date_parser=parser)

# Get test data values into objects
testPrice = testData.iloc[:, [7]].fillna(method='ffill').values
testOpen = testData.iloc[:, [1]].fillna(method='ffill').values
testHigh = testData.iloc[:, [2]].fillna(method='ffill').values
testLow = testData.iloc[:, [3]].fillna(method='ffill').values
testClose = testData.iloc[:, [4]].fillna(method='ffill').values

# Converting the dataframes into lists
historyPrice = [x for x in trainPrice]
historyOpen = [x for x in trainOpen]
historyHigh = [x for x in trainHigh]
historyLow = [x for x in trainLow]
historyClose = [x for x in trainClose]

testLen = len(testPrice)
predictions = list()

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
    predictions.append(predict)
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
    #Add the testValues to history for 
# Save the output to a csv file
predictionsDf = DataFrame(predictions)
predictionsDf.to_csv('arimaOutput.csv', sep=',')

