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
*Setting up the files*
```python
# Files to process
"""
Modify the locations below as per your directory struture.
"""
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"

# Loading the csv tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header=None, names=["Tickers"])
```

*Data preprocessing functions*
```python
def prep_data(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total current assets", "Total non-current assets", "Total current liabilities", "Total non-current liabilities"]]
    balance_sheet.rename(index = {
        "Total current assets": "crr_asst",
        "Total non-current assets": "ncrr_asst",
        "Total current liabilities": "crr_libt",
        "Total non-current liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df = (df - df.shift(1))/abs(df.shift(1))
    df *= 100
    df.dropna(inplace = True)
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]]
#     print(df)
    df.to_csv(prep_dir + ticker + ".csv")
    return

# prep_data("AES")
```

```python
# the companies for which balance sheet without current and non-current assets and liabilities
def prep_data_2(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total assets", "Total liabilities"]]
    balance_sheet.rename(index = {
        "Total assets": "ncrr_asst",
        "Total liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df = (df - df.shift(1))/abs(df.shift(1))
    df *= 100
    df.dropna(inplace = True)
    df.insert(0, "crr_asst", 0)
    df.insert(0, "crr_libt", 0)
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]]
    df.to_csv(prep_dir + ticker + ".csv")
    return
```

*Preprocessing the data*
```python
counter = 0
not_done_set = set()
for ticker in ticker_list_df["Tickers"]:
    counter += 1
    try:
        print("Pre Processing Data for: " + ticker + "(" + str(counter) + "/"+ str(len(ticker_list_df)) +"; Failed(" + str(len(not_done_set)) + "))")
        prep_data(ticker)
    except:
        try:
            print("--------------------------------------------------")
            prep_data_2(ticker)
        except Exception as e:
            print(e)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            not_done_set.add(ticker)
```

# Credit Rating Model

### Multivariate Parallel LSTM RNN Model
![Abstract RNN Model](images/abstract-rnn-model.png)

*Setting up the files*
```python
# Files to process
"""
Modify the locations below as per your directory struture.
"""
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"

# Loading the csv tickers, train_tickers and test_tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header = None, names = ["Tickers"])
train_tickers_df = pd.read_csv(root_dir + "train_tickers.csv", header = None, names = ["Train Tickers"])
test_tickers_df = pd.read_csv(root_dir + "test_tickers.csv", header = None, names = ["Test Tickers"])

train_set = set(train_tickers_df["Train Tickers"].tolist())
test_set = set(test_tickers_df["Test Tickers"].tolist())
```

*Fetching and splitting*
```python
def in_out_split(data, n_steps):
    for i in range(1, len(data)):
        end_ix = i + n_steps
        if(end_ix > len(data) - 1):
            break
            
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
        
        if ticker in train_set:
            X_train.append(seq_x)
            y_train.append(seq_y)
        else:
            X_test.append(seq_x)
            y_test.append(seq_y)
            

def read_data_from_file(ticker):
    df = pd.read_csv(prep_dir + ticker + ".csv")
    return df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]].values
    
X_train, y_train = list(), list()
X_test, y_test = list(), list()

n_steps = 2

counter = 0
for ticker in ticker_list_df["Tickers"]:
    counter += 1
    print("Fetching data for: " + ticker + "(" + str(counter) + "/"+ str(len(ticker_list_df)) + ")")
    data = read_data_from_file(ticker)
    in_out_split(data, n_steps)
```

*Model Creation*
```python
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(100, activation = "relu", return_sequences = True, input_shape = (n_steps, n_features)))
model.add(LSTM(100, activation = "relu"))
model.add(Dense(n_features))

model.compile(optimizer = "adam", loss = "mse")
model.fit(X_train, y_train, epochs = 100, batch_size = 32, shuffle = True, verbose = 1)

model.save(root_dir + "bondai_model_1.1.h5")
```

### Model Analysis and Prediction
*Setting up files*
```python
# Files to process
"""
Modify the locations below as per your directory struture.
"""
sp_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/"
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"
model_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/model-1/"


# Loading the csv tickers, train_tickers and test_tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header = None, names = ["Tickers"])
train_tickers_df = pd.read_csv(root_dir + "train_tickers.csv", header = None, names = ["Train Tickers"])
test_tickers_df = pd.read_csv(root_dir + "test_tickers.csv", header = None, names = ["Test Tickers"])
```

*Fetching actual data*
```python
def get_actual_data(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total current assets", "Total non-current assets", "Total current liabilities", "Total non-current liabilities"]]
    balance_sheet.rename(index = {
        "Total current assets": "crr_asst",
        "Total non-current assets": "ncrr_asst",
        "Total current liabilities": "crr_libt",
        "Total non-current liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]][-1:]
    df["Ticker"] = ticker
    df.set_index("Ticker", inplace = True)
#     print(df)
    return df
```

```python
# the companies for which balance sheet without current and non-current assets and liabilities
def get_actual_data_2(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total assets", "Total liabilities"]]
    balance_sheet.rename(index = {
        "Total assets": "ncrr_asst",
        "Total liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df["Ticker"] = ticker
    df.set_index("Ticker", inplace = True)
    df.insert(0, "crr_asst", 0)
    df.insert(0, "crr_libt", 0)
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]][-1:]
    return df
    
df = pd.DataFrame(columns = ["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"])
df.index.name = "Ticker"
counter = 0
for ticker in test_tickers_df["Test Tickers"]:
    counter += 1
    print(str(counter) + ". Getting data for: " + ticker)
    try:
        df = df.append(get_actual_data(ticker))
    except:
        df = df.append(get_actual_data_2(ticker))
        
df = df.dropna()
df

df.to_csv(model_dir + "actual_data.csv")
```

```python
def get_prev_data(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total current assets", "Total non-current assets", "Total current liabilities", "Total non-current liabilities"]]
    balance_sheet.rename(index = {
        "Total current assets": "crr_asst",
        "Total non-current assets": "ncrr_asst",
        "Total current liabilities": "crr_libt",
        "Total non-current liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]][-2:-1]
    df["Ticker"] = ticker
    df.set_index("Ticker", inplace = True)
#     print(df)
    return df

# get_prev_data("HSIC")
```

```python
# the companies for which balance sheet without current and non-current assets and liabilities
def get_prev_data_2(ticker):
    # INCOME STATEMENT
    income_statement = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "income_statement")
    income_statement = income_statement.loc[["Gross Profit", "Operating Income", "Net Income"]]
    income_statement.rename(index = {
        "Gross Profit": "gross_profit",
        "Operating Income": "op_income",
        "Net Income": "net_income"
    }, inplace = True)
    
    # BALANCE SHEET
    balance_sheet = pd.read_excel(data_dir + ticker + ".xlsx", sheet_name = "balance_sheet")
    balance_sheet = balance_sheet.loc[["Total assets", "Total liabilities"]]
    balance_sheet.rename(index = {
        "Total assets": "ncrr_asst",
        "Total liabilities": "ncrr_libt"
    }, inplace = True)
    
    df = pd.concat([income_statement, balance_sheet])
    df = df[df.columns[::-1]]
    df = df.transpose()
    df["Ticker"] = ticker
    df.set_index("Ticker", inplace = True)
    df.insert(0, "crr_asst", 0)
    df.insert(0, "crr_libt", 0)
    df = df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]][-2:-1]
    return df
```

```python
df2 = pd.DataFrame(columns = ["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"])
df2.index.name = "Ticker"
counter = 0
for ticker in test_tickers_df["Test Tickers"]:
    counter += 1
    print(str(counter) + ". Getting data for: " + ticker)
    try:
        df2 = df2.append(get_prev_data(ticker))
    except:
        df2 = df2.append(get_prev_data_2(ticker))
        
df2 = df2.dropna()
df2

df2.to_csv(model_dir + "prev_data.csv")
```

*Preparing test data*
```python
def prepare_test_data(data, n_steps):
    for i in range(1, len(data)):
        end_ix = i + n_steps
        if(end_ix > len(data) - 1):
            seq_x = data[i-1:end_ix-1, :]
            return seq_x
            

def read_data_from_file(ticker):
    df = pd.read_csv(prep_dir + ticker + ".csv")
    return df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]].values

```

*Prediction*
```python
# Load trained model
from keras.models import load_model
model = load_model(sp_dir + "bondai_model_1.1.h5")

df3 = pd.DataFrame(columns = ["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"])
df3.index.name = "Ticker"
test_df = pd.DataFrame(columns = ["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"])
test_df.index.name = "Ticker"
counter = 0
for ticker in test_tickers_df["Test Tickers"]:
    counter += 1
    print(str(counter) + ". Predicting values for: " + ticker)
    test_array = prepare_test_data(read_data_from_file(ticker), 2)
    test_input = test_array.reshape(1, 2, 7)
    pred_array = model.predict(test_input, verbose = 1)
    df_temp = pd.DataFrame(pred_array.tolist(), columns = ["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"])
    df_temp["Ticker"] = ticker
    df_temp.set_index("Ticker", inplace = True)
    test_df = test_df.append(df_temp)
    df_temp = df_temp.loc[ticker]*df2.loc[ticker]
    df3 = df3.append(df_temp)
   
# Saving the predictions to a csv file
df3.to_csv(model_dir + "pred_data.csv")
```

### Credit Rating
*Setting up files*
```python
# Files to process
"""
Modify the locations below as per your directory struture.
"""
sp_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/"
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"
model_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/model-1/"

# Loading the csv files
import pandas as pd
actual_df = pd.read_csv(model_dir + "actual_data.csv")
pred_df = pd.read_csv(model_dir + "pred_data.csv")
```

*Setting up the dataframes*
```python
actual_df["act_short_term"] = actual_df["crr_asst"]/actual_df["crr_libt"]
actual_df["act_long_term"] = actual_df["ncrr_asst"]/actual_df["ncrr_libt"]
actual_df["act_overall"] = (actual_df["crr_asst"] + actual_df["ncrr_asst"])/(actual_df["crr_libt"] + actual_df["ncrr_libt"])

pred_df["pred_short_term"] = pred_df["crr_asst"]/pred_df["crr_libt"]
pred_df["pred_long_term"] = pred_df["ncrr_asst"]/pred_df["ncrr_libt"]
pred_df["pred_overall"] = (pred_df["crr_asst"] + pred_df["ncrr_asst"])/(pred_df["crr_libt"] + pred_df["ncrr_libt"])

actual_rating = actual_df[["Ticker", "act_short_term", "act_long_term", "act_overall"]]
actual_rating.set_index("Ticker", inplace = True)

pred_rating = pred_df[["Ticker", "pred_short_term", "pred_long_term", "pred_overall"]]
pred_rating.set_index("Ticker", inplace = True)

rating_df = pd.concat([actual_rating, pred_rating], axis = 1)
rounded_df = round(rating_df)
rounded_df = pd.read_csv(model_dir + "credit_ratings_1.0.csv")
```

*Graphical Visualization*
```python
import matplotlib.pyplot as plt
```

*Logn term credit ratings*
![Long Term Credit Ratings]()
```python
plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_long_term"], "b", label = "actual")
plt.plot(rounded_df["pred_long_term"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Long Term Credit Rating")
plt.show()

plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_long_term"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_long_term"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Long Term Credit Rating")
plt.show()

plt.hist([rounded_df["act_long_term"], rounded_df["pred_long_term"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Long Term Credit Rating")
plt.show()
```

*Overall credit ratings*
![Overall Credit Ratings]()
```python
plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_overall"], "b", label = "actual")
plt.plot(rounded_df["pred_overall"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Overall Credit Rating")
plt.show()

plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_overall"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_overall"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Overall Credit Rating")
plt.show()

plt.hist([rounded_df["act_overall"], rounded_df["pred_overall"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Overall Credit Rating")
plt.show()
```

*Short term credit ratings*
![Short Term Credit Ratings]()
```python
plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_short_term"], "b", label = "actual")
plt.plot(rounded_df["pred_short_term"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Short Term Credit Rating")
plt.show()

plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_short_term"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_short_term"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Short Term Credit Rating")
plt.show()

plt.hist([rounded_df["act_short_term"], rounded_df["pred_short_term"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Short Term Credit Rating")
plt.show()
```






# Financial Sentiment Analysis Model

### Financial Sentiment Intensity Prediction

### Stockprice Timeline Analysis

### Cummulative Sentiment vs Stockprice Fluctuations

### Random Forest Model 
