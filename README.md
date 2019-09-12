# Stock Analyzer
Stock Analysis webapp with Pandas, Flask, Scikit-Learn and beautiful AmCharts


Usage:

Open terminal:

```
$ pip install -r requirements.txt
$ export FLASK_APP=app.py
$ flask run
```

## WebApp Usage:

### Once in webapp input stock symbol you want to analyze, then select start date and end date, be weary of date selection for every stock. App is using Pandas_DataReader to retrieve the data. After hitting submit, you'll be presented with a beautiful AmCharts graph.

## TODO:

#### * Create APIs for data processing instead of running the whole logica app for the data process
#### * ~~Save the trained models in ONNX or other type of standard~~ Saved in pickle for now
#### * Create other views for data analysis
#### * Make the top bar work correctly, returning the data to the ORIGIN request


***This is beta software so remember everything might probably come crashing down...

Made with ❤ by repollo from nori-labs.com