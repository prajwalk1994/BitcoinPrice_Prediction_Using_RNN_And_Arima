from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

def parser(x):
    dateout = []
    for time in x:
        dateout.append(datetime.fromtimestamp(int(time)))
    return dateout


series = read_csv('CMPE-256-Large-Scale-Analytics-/data/bitstamp.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

print(series.head(10))