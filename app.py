from flask import Flask,render_template,request,jsonify
from random import sample
import gatherer
import company

app = Flask(__name__)

start = ""
end = ""

@app.route('/')
def index():
	return render_template('main.html')

@app.route('/data',methods=["POST", 'GET'])
def data():
	if request.method=='POST':
		symbol = request.form['search']
		source = request.form['sourcery']
		start = request.form['trip-start']
		end = request.form['trip-end']
		dt, dd, mav, rets = gatherer.data(symbol, source, start, end)
		comp_name = company.get_symbol(symbol)
		return render_template('chart.html', stock_date=dt, stock_data=dd, mav=mav, rets=rets, company=comp_name, start=start, end=end)

if __name__ == '__main__':
	app.run(debug=1)