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


traindata = read_csv('/train.csv')
testdata  = read_csv('/test.csv')


train = traindata.iloc[:, [6]].fillna(method ='ffill')
test =  traindata.iloc[:, [6]].fillna(method ='ffill')



print(len(train))
print(len(test))

#Model the train Set

model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

# Make predictions
prediction = model_fit.predict(start= len(train), end=len(train)+len(test)-1, dynamic=False)
outputs = list()
for i in range(len(prediction)):
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
