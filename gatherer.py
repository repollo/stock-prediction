import pandas as pd
import datetime
import pandas_datareader.data as web

def data(arg,kwargs, start, end):

	try:
		#Try to obtain data
		df = web.DataReader(arg, kwargs, start, end)
		#Get all closing values
		close_px = df['Adj Close']
		#Create moving avergae values
		mavg = close_px.rolling(window=100).mean()

		#Calculate rets
		rets = close_px / close_px.shift(1) - 1

		#Provide data to Flask app
		# return close_px.to_json(orient='index',date_format='iso'), close_px.to_list(), mavg.to_list()
		return close_px.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), close_px.to_list(), mavg.to_list(), rets.to_list()

	#If any error, provide back to flask app, although it does not work properly.
	except TypeError as e:
		return e
	except NameError as e:
		return e
	except Exception as e:
		return e
	except RemoteDataError as e:
		return e
