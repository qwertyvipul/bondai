# BondAI
An AI Based Bond Credit Rating System


# Tech Stack
### Programming Languages -
1. Python

### Development Environment -
1. Google Colaboratory
2. IBM Watson
3. Jupyter Notebooks
4. Python 3.x

### Python Libraries
1. urllib
2. pandas
3. numpy
4. matplotlib
5. tensorflow
6. keras
7. bs4
8. yahoo-finance
9. sklearn

# Dataset Framework



### Data Accqusition
*Setting up the environment*
```python
# Mount Google Drive
"""
No need to execute this block when working on local system.
"""
from google.colab import drive
drive.mount("/content/vdrive", force_remount = True)
```

*Setting up the meta files*
```python
# Files to process
"""
Modify the locations below as per your directory struture.
"""
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
do_file = root_dir + "do_file.csv"
done_file = root_dir + "done_file.csv"
not_done_file = root_dir + "not_done_file.csv"

# Reading the files
import pandas as pd
do_df = pd.read_csv(do_file, header=None, names=["Tickers"])
done_df = pd.read_csv(done_file, header=None, names=["Tickers"])
not_done_df = pd.read_csv(not_done_file, header=None, names=["Tickers"])

do_set = set(do_df["Tickers"].tolist())
done_set = set(done_df["Tickers"].tolist())
not_done_set = set(not_done_df["Tickers"].tolist())
```

*Stockrow Downloader*
```python
# URL Paths for Stockrow Website
stockrow_url_paths = {
    'company': 'https://stockrow.com/api/companies/',
    'annual': {
        'income-statement': '/financials.xlsx?dimension=MRY&section=Income%20Statement&sort=desc',
        'balance-sheet': '/financials.xlsx?dimension=MRY&section=Balance%20Sheet&sort=desc',
        'cashflow-statement': '/financials.xlsx?dimension=MRY&section=Cash%20Flow&sort=desc',
        'metrics': '/financials.xlsx?dimension=MRY&section=Metrics&sort=desc',
        'growth': '/financials.xlsx?dimension=MRY&section=Growth&sort=desc'
    } 
}

# Stockrow Downloader
import requests
def stockrow_download(ticker):
    income_statement = pd.read_excel(stockrow_url_paths['company'] + ticker + stockrow_url_paths['annual']["income-statement"], engine="xlrd")
    balance_sheet = pd.read_excel(stockrow_url_paths['company'] + ticker + stockrow_url_paths['annual']["balance-sheet"], engine="xlrd")
    cashflow_statement = pd.read_excel(stockrow_url_paths['company'] + ticker + stockrow_url_paths['annual']["cashflow-statement"], engine="xlrd")
    metrics = pd.read_excel(stockrow_url_paths['company'] + ticker + stockrow_url_paths['annual']["metrics"], engine="xlrd")
    growth = pd.read_excel(stockrow_url_paths['company'] + ticker + stockrow_url_paths['annual']["growth"], engine="xlrd")
    return income_statement, balance_sheet, cashflow_statement, metrics, growth
```

*Yahoo finance Downloader*
```python
# Modified Get Yahoo Quotes Script by Brad Luicas

__author__ = "Brad Luicas"
__copyright__ = "Copyright 2017, Brad Lucas"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brad Lucas"
__email__ = "brad@beaconhill.com"
__status__ = "Production"

import re
import sys
import time
import datetime
# import requests


def split_crumb_store(v):
    return v.split(':')[2].strip('"')


def find_crumb_store(lines):
    # Looking for
    # ,"CrumbStore":{"crumb":"9q.A4D1c.b9
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")


def get_cookie_value(r):
    return {'B': r.cookies['B']}


def get_page_data(symbol):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
    r = requests.get(url)
    cookie = get_cookie_value(r)

    # Code to replace possible \u002F value
    # ,"CrumbStore":{"crumb":"FWP\u002F5EFll3U"
    # FWP\u002F5EFll3U
    lines = r.content.decode('unicode-escape').strip(). replace('}', '\n')
    return cookie, lines.split('\n')


def get_cookie_crumb(symbol):
    cookie, lines = get_page_data(symbol)
    crumb = split_crumb_store(find_crumb_store(lines))
    return cookie, crumb


def get_data(symbol, start_date, end_date, cookie, crumb):
    # filename = '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie)
    # with open (filename, 'wb') as handle:
    #     for block in response.iter_content(1024):
    #         handle.write(block)
    return response


def get_now_epoch():
    # @see https://www.linuxquestions.org/questions/programming-9/python-datetime-to-epoch-4175520007/#post5244109
    return int(time.time())


def download_quotes(symbol):
    start_date = 0
    end_date = get_now_epoch()
    cookie, crumb = get_cookie_crumb(symbol)
    historical_prices = get_data(symbol, start_date, end_date, cookie, crumb)
    return pd.read_csv(io.StringIO(historical_prices.content.decode('utf-8')))
```

*Fetching the data*
```python
import os
import io
def main():
    counter = 0
    total = len(do_set)
    
    print(do_set)
    for ticker in do_set.copy():
        counter = counter + 1
        try:
            print("Downloading data for: " + ticker + "(" + str(counter) + "/" + str(total) +"); Failed(" + str(len(not_done_set)) + ")")
            income_statement, balance_sheet, cashflow_statement, metrics, growth = stockrow_download(ticker)
            historical_prices = download_quotes(ticker)
            with pd.ExcelWriter(root_dir + "raw/" + ticker + '.xlsx') as writer:
                historical_prices.to_excel(writer, sheet_name="historical_prices")
                balance_sheet.to_excel(writer, sheet_name="balance_sheet")
                income_statement.to_excel(writer, sheet_name="income_statement")
                cashflow_statement.to_excel(writer, sheet_name="cashflow_statement")
                metrics.to_excel(writer, sheet_name="metrics")
                growth.to_excel(writer, sheet_name="growth")
        except:
            not_done_set.add(ticker)
            do_set.discard(ticker)
            pd.DataFrame(list(not_done_set)).to_csv(not_done_file, header=None, index=False)
            pd.DataFrame(list(do_set)).to_csv(do_file, header=None, index=False)
            continue
            
            
        done_set.add(ticker)
        do_set.discard(ticker)
        pd.DataFrame(list(done_set)).to_csv(done_file, header=None, index=False)
        pd.DataFrame(list(not_done_set)).to_csv(not_done_file, header=None, index=False)
        pd.DataFrame(list(do_set)).to_csv(do_file, header=None, index=False)
```


### Data preprocessing


# Credit Rating Model

### Multivariate Parallel LSTM RNN Model
![Abstract RNN Model](images/abstract-rnn-model.png)

### Model Analysis

### Credit Rating


# Financial Sentiment Analysis Model

### Financial Sentiment Intensity Prediction

### Stockprice Timeline Analysis

### Cummulative Sentiment vs Stockprice Fluctuations

### Random Forest Model 
