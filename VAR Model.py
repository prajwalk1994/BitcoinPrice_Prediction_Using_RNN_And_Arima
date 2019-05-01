from pandas import read_csv
from pandas import datetime
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


traindata = read_csv('train.csv', header=0,parse_dates=[0], index_col=0,squeeze=True)
testdata  = read_csv('test.csv', header=0,parse_dates=[0], index_col=0,squeeze=True)

traindata['date'] = to_datetime(traindata['Timestamp'],unit='s').dt.date
testdata['date'] = to_datetime(testdata['Timestamp'],unit='s').dt.date

trainDatagroup=traindata.groupby('date')
trainprice=trainDatagroup['Weighted_Price'].mean().values
testDatagroup=testdata.groupby('date')
testprice=testDatagroup['Weighted_Price'].mean().values

#pricetrain = traindata.iloc[:, [6]].fillna(method ='ffill')
#pricetest= testdata.iloc[:, [6]].fillna(method ='ffill')

print(testdata)
#print(test)


#series = Series.from_csv('bitmap.csv', header=0)
# split dataset
#X = price.head(100).values

#train, test = X[0:len(X)-10], X[len(X)-10:]
#print(train[:10])

#series = read_csv('bitstamp.csv', header=0,parse_dates=[0], index_col=0, squeeze=True)
#price = series.iloc[:, [6]].fillna(method ='ffill')
# split dataset
train = pricetrain.tail(5000).values
test=   pricetest.head(1000).values



#train, test = X[1:len(X)-1000], X[len(X)-1000:]
print(len(train))
print(len(test))
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
print(window)
coef = model_fit.params
print(coef)
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()

#from statsmodels.tsa.vector_ar.var_model import VAR

#model = VAR(train)
#model_fit = model.fit()

# make prediction on validation
#prediction = model_fit.forecast(model_fit.y, steps=len(valid))
#print(prediction)
# print(len(test))
#temp = [x for x in train]
# train autoregression
#model = AR(temp)
#model_fit = model.fit()
#print('Lag: %s' % model_fit.k_ar)
#print('Coefficients: %s' % model_fit.params)
# make predictions
#predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
#for i in range(len(predictions)):
#	print('predicted=%f, expected=%f' % (predictions[i], test[i]))