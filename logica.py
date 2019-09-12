import math
import pandas as pd
import datetime
import math
import numpy as np
import pandas_datareader.data as web
import pickle
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, BayesianRidge, LassoLars, OrthogonalMatchingPursuit, ARDRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c

######DISCLAIMER! THIS VIEW IS HORRIFIC, WATCH WITH YOUR OWN RISK OF BLEEDING EYES AND TURNING EYES BACK!, YOU'VE BEEN WARNED!#######

def task1(df):

	try:
		#Get all closing values
		close_px = df['Adj Close']

		#Create moving avergae values
		mavg = close_px.rolling(window=100).mean()

		#Calculate rets
		rets = close_px / close_px.shift(1) - 1

		#Provide data to Flask app
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

def task2(data):

	df = data

	dfreg = df.loc[:,['Adj Close','Volume']]
	dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
	dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

	# Drop missing value
	dfreg.fillna(value=-99999, inplace=True)
	# We want to separate 1 percent of the data to forecast
	forecast_out = int(math.ceil(0.01 * len(dfreg)))
	# Separating the label here, we want to predict the AdjClose
	forecast_col = 'Adj Close'
	dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
	X = np.array(dfreg.drop(['label'], 1))
	# Scale the X so that everyone can have the same distribution for linear regression
	X = preprocessing.scale(X)
	# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]
	# Separate label and identify it as y
	y = np.array(dfreg['label'])
	y = y[:-forecast_out]
	
	#Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	##################
	##################
	##################


	# Linear regression
	clfreg = LinearRegression(n_jobs=-1)
	# 1 - First save the models to local device in models folder
	# filename = 'models/clfreg_model.sav'
	# pickle.dump(clfreg, open(filename, 'wb'))

	# 2 - load the models from disk onces first instruction is done once.
	# clfreg = pickle.load(open(filename, 'rb'))
	clfreg.fit(X_train, y_train)


	# Quadratic Regression 2
	clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
	#Save model to a pickle
	# filename1 = 'models/clfpoly2_model.sav'
	# pickle.dump(clfpoly2, open(filename1, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfpoly2 = pickle.load(open(filename1, 'rb'))
	clfpoly2.fit(X_train, y_train)


	# Quadratic Regression 3
	clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
	#Save model to a pickle
	# filename2 = 'models/clfpoly3_model.sav'
	# pickle.dump(clfpoly3, open(filename2, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfpoly3 = pickle.load(open(filename2, 'rb'))
	clfpoly3.fit(X_train, y_train)


	# KNN Regression
	clfknn = KNeighborsRegressor(n_neighbors=2)
	#Save model to a pickle
	# filename3 = 'models/clfknn_model.sav'
	# pickle.dump(clfknn, open(filename3, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfknn = pickle.load(open(filename3, 'rb'))
	clfknn.fit(X_train, y_train)


	# Lasso Regression
	clflas = Lasso()
	#Save model to a pickle
	# filename4 = 'models/clflas_model.sav'
	# pickle.dump(clflas, open(filename4, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clflas = pickle.load(open(filename4, 'rb'))
	clflas.fit(X_train, y_train)


	# Multitask Lasso Regression
	# clfmtl = MultiTaskLasso(alpha=1.)
	# clfmtl.fit(X_train, y_train).coef_


	# Bayesian Ridge Regression
	clfbyr = BayesianRidge()
	clfbyr.fit(X_train, y_train)
	#Save model to a pickle
	# filename5 = 'models/clfbyr_model.sav'
	# pickle.dump(clfbyr, open(filename5, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfbyr = pickle.load(open(filename5, 'rb'))


	# Lasso LARS Regression
	clflar = LassoLars(alpha=.1)
	clflar.fit(X_train, y_train)
	#Save model to a pickle
	# filename6 = 'models/clflar_model.sav'
	# pickle.dump(clflar, open(filename6, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clflar = pickle.load(open(filename6, 'rb'))


	# Orthogonal Matching Pursuit Regression
	clfomp = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
	clfomp.fit(X_train, y_train)
	#Save model to a pickle
	# filename7 = 'models/clfomp_model.sav'
	# pickle.dump(clfomp, open(filename7, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfomp = pickle.load(open(filename7, 'rb'))


	# Automatic Relevance Determination Regression
	clfard = ARDRegression(compute_score=True)
	clfard.fit(X_train, y_train)
	#Save model to a pickle
	# filename8 = 'models/clfard_model.sav'
	# pickle.dump(clfard, open(filename8, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfard = pickle.load(open(filename8, 'rb'))


	# Logistic Regression
	# clflgr = linear_model.LogisticRegression(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6), warm_start=True)
	# coefs_ = []
	# for c in cs:
	#   clflgr.set_params(C=c)
	#   clflgr.fit(X_train, y_train)
	#   coefs_.append(clflgr.coef_.ravel().copy())


	#SGD Regression
	clfsgd = SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)
	clfsgd.fit(X_train, y_train)
	#Save model to a pickle
	# filename9 = 'models/clfsgd_model.sav'
	# pickle.dump(clfsgd, open(filename9, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfsgd = pickle.load(open(filename9, 'rb'))


	##################
	##################
	##################


	#Create confindence scores
	confidencereg = clfreg.score(X_test, y_test)
	confidencepoly2 = clfpoly2.score(X_test,y_test)
	confidencepoly3 = clfpoly3.score(X_test,y_test)
	confidenceknn = clfknn.score(X_test, y_test)
	confidencelas = clflas.score(X_test, y_test)
	# confidencemtl = clfmtl.score(X_test, y_test)
	confidencebyr = clfbyr.score(X_test, y_test)
	confidencelar = clflar.score(X_test, y_test)
	confidenceomp = clfomp.score(X_test, y_test)
	confidenceard = clfard.score(X_test, y_test)
	confidencesgd = clfsgd.score(X_test, y_test)

	# results
	print('The linear regression confidence is:',confidencereg*100)
	print('The quadratic regression 2 confidence is:',confidencepoly2*100)
	print('The quadratic regression 3 confidence is:',confidencepoly3*100)
	print('The knn regression confidence is:',confidenceknn*100)
	print('The lasso regression confidence is:',confidencelas*100)
	# print('The lasso regression confidence is:',confidencemtl*100)
	print('The Bayesian Ridge regression confidence is:',confidencebyr*100)
	print('The Lasso LARS regression confidence is:',confidencelar*100)
	print('The OMP regression confidence is:',confidenceomp*100)
	print('The ARD regression confidence is:',confidenceard*100)
	print('The SGD regression confidence is:',confidencesgd*100)

	#Create new columns
	forecast_reg = clfreg.predict(X_lately)
	forecast_pol2 = clfpoly2.predict(X_lately)
	forecast_pol3 = clfpoly3.predict(X_lately)
	forecast_knn = clfknn.predict(X_lately)
	forecast_las = clflas.predict(X_lately)
	forecast_byr = clfbyr.predict(X_lately)
	forecast_lar = clflar.predict(X_lately)
	forecast_omp = clfomp.predict(X_lately)
	forecast_ard = clfard.predict(X_lately)
	forecast_sgd = clfsgd.predict(X_lately)

	#Process all new columns data
	dfreg['Forecast_reg'] = np.nan

	last_date = dfreg.iloc[-1].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)

	for i in forecast_reg:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns))]
	    dfreg['Forecast_reg'].loc[next_date] = i
	    
	dfreg['Forecast_pol2'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_pol2:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_pol2'].loc[next_date] = i

	dfreg['Forecast_pol3'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_pol3:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_pol3'].loc[next_date] = i
	    
	dfreg['Forecast_knn'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_knn:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_knn'].loc[next_date] = i
	        
	dfreg['Forecast_las'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_las:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_las'].loc[next_date] = i
	    
	dfreg['Forecast_byr'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_byr:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_byr'].loc[next_date] = i
	    
	dfreg['Forecast_lar'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_lar:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_lar'].loc[next_date] = i
	    
	dfreg['Forecast_omp'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_omp:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_omp'].loc[next_date] = i
	    
	dfreg['Forecast_ard'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_ard:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_ard'].loc[next_date] = i
	    
	dfreg['Forecast_sgd'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_sgd:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_sgd'].loc[next_date] = i

	return dfreg.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), dfreg['Adj Close'].to_list(), dfreg['Forecast_reg'].to_list(), dfreg['Forecast_pol2'].to_list(), dfreg['Forecast_pol3'].to_list(), dfreg['Forecast_knn'].to_list(), dfreg['Forecast_las'].to_list(), dfreg['Forecast_byr'].to_list(), dfreg['Forecast_lar'].to_list(), dfreg['Forecast_omp'].to_list(), dfreg['Forecast_ard'].to_list(), dfreg['Forecast_sgd'].to_list()








