from flask import Flask,render_template,request,jsonify
from random import sample
import pandas as pd
import gatherer
import company
import logica

app = Flask(__name__)

symbol = ""
start = ""
end = ""
data = pd.DataFrame()
comp_name = ""

@app.route('/')
def index():
	return render_template('main.html')

@app.route('/data',methods=["POST", 'GET'])
def data():

	global symbol
	global start
	global end
	global data
	global comp_name

	if request.method=='POST':
		print(request)
		# if symbol != request.form['search']:
		symbol = request.form['search']
		source = request.form['sourcery']
		start = request.form['trip-start']
		end = request.form['trip-end']

		data = gatherer.data(symbol, source, start, end)
		comp_name = company.get_symbol(symbol)
		return chart1()

@app.route('/chart1')
def chart1():

	global start
	global end
	global data
	global comp_name

	dt, dd, mav, rets = logica.task1(data)
	print(type(dd))
	return render_template('chart1.html', stock_date=dt, stock_data=dd, mav=mav, company=comp_name, start=start, end=end)

@app.route('/chart2')
def chart2():

	global start
	global end
	global data
	global comp_name

	dt, dd, mav, rets = logica.task1(data)
	return render_template('chart2.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

@app.route('/chart3')
def chart3():

	global start
	global end
	global data
	global comp_name

	dt, dd, reg, pol2, pol3, knn, las, byr, lar, omp, ard, sgd = logica.task2(data)
	return render_template('chart3.html', stock_date=dt, stock_data=dd, reg=reg, pol2=pol2, pol3=pol3, knn=knn, las=las, byr=byr, lar=lar, omp=omp, ard=ard, sgd=sgd, company=comp_name, start=start, end=end)

# @app.route('/chart4')
# def chart4():

# 	global start
# 	global end
# 	global data
# 	global comp_name
	
# 	dt, dd, mav, rets = logica.task1(data)
# 	return render_template('chart4.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

# @app.route('/chart5')
# def chart5():

# 	global start
# 	global end
# 	global data
# 	global comp_name

# 	dt, dd, mav, rets = logica.task1(data)
# 	return render_template('chart5.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

# @app.route('/chart6')
# def chart6():

# 	global start
# 	global end
# 	global data
# 	global comp_name

# 	dt, dd, mav, rets = logica.task1(data)
# 	return render_template('chart6.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

# @app.route('/chart7')
# def chart7():

# 	global start
# 	global end
# 	global data
# 	global comp_name

# 	dt, dd, mav, rets = logica.task1(data)
# 	return render_template('chart7.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

if __name__ == '__main__':
	app.run(debug=1)




