import pandas as pd
import datetime
import pandas_datareader.data as web

def data(arg,kwargs, start, end):

	try:
		#Try to obtain data
		df = web.DataReader(arg, kwargs, start, end)
		
		return df
	#If any error, provide back to flask app, although it does not work properly.
	except TypeError as e:
		return e
	except NameError as e:
		return e
	except Exception as e:
		return e
	except RemoteDataError as e:
		return e
