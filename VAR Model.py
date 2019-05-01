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


series = read_csv('bitstamp.csv', header=0,
                  parse_dates=[0], index_col=0, squeeze=True)
price = series.iloc[:, [6]].fillna(method ='ffill')


#series = Series.from_csv('bitmap.csv', header=0)
# split dataset
#X = price.head(100).values

#train, test = X[0:len(X)-10], X[len(X)-10:]
#print(train[:10])

series = read_csv('bitstamp.csv', header=0,parse_dates=[0], index_col=0, squeeze=True)
price = series.iloc[:, [6]].fillna(method ='ffill')
# split dataset
X = price.values
print(len(X))
train, test = X[1:len(X)-1000], X[len(X)-1000:]
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
	predictions.append(yhat+17)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
	