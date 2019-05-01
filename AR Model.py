from pandas import read_csv
from pandas import datetime, DataFrame
from pandas import to_datetime
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error

def parser(x):
    dateout = []
    for time in x:
        dateout.append(datetime.fromtimestamp(int(time)))
    return dateout


traindata = read_csv('train.csv')
testdata  = read_csv('test.csv')

traindata['date'] = to_datetime(traindata['Timestamp'],unit='s').dt.date
testdata['date'] = to_datetime(testdata['Timestamp'],unit='s').dt.date

trainDatagroup=traindata.groupby('date')
train=trainDatagroup['Weighted_Price'].mean().values
testDatagroup=testdata.groupby('date')
test=testDatagroup['Weighted_Price'].mean().values
timestamps = testDatagroup['date'].unique().values


print(len(train))
print(len(test))



model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start= len(train), end=len(train)+len(test)-1, dynamic=False)
outputs = list()
for i in range(len(predictions)):
    predict = list()
    predict.append(timestamps[i][0].strftime('%m-%d-%y'))
    predict.append(predictions[i])
    predict.append(test[i])
    outputs.append(predict)
# 	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
predictionsDf = DataFrame(outputs)
predictionsDf.to_csv('varOutput.csv', sep=',')
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
