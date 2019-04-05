
# coding: utf-8

# In[2]:


# Mount Google Drive
"""
No need to execute this block when working on local system.
"""
from google.colab import drive
drive.mount("/content/vdrive", force_remount = True)


# In[ ]:


# Files to process
"""
Modify the locations below as per your directory struture.
"""
sp_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/"
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"
model_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/model-1/"


# In[ ]:


# Loading the csv tickers, train_tickers and test_tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header = None, names = ["Tickers"])
train_tickers_df = pd.read_csv(root_dir + "train_tickers.csv", header = None, names = ["Train Tickers"])
test_tickers_df = pd.read_csv(root_dir + "test_tickers.csv", header = None, names = ["Test Tickers"])


# In[ ]:


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


# In[ ]:


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


# In[100]:


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


# In[104]:


df = df.dropna()
df


# In[ ]:


df.to_csv(model_dir + "actual_data.csv")


# In[ ]:


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


# In[ ]:


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


# In[121]:


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


# In[122]:


df2 = df2.dropna()
df2


# In[ ]:


df2.to_csv(model_dir + "prev_data.csv")


# In[ ]:


def prepare_test_data(data, n_steps):
    for i in range(1, len(data)):
        end_ix = i + n_steps
        if(end_ix > len(data) - 1):
            seq_x = data[i-1:end_ix-1, :]
            return seq_x


# In[ ]:


def read_data_from_file(ticker):
    df = pd.read_csv(prep_dir + ticker + ".csv")
    return df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]].values


# In[ ]:


# Load trained model
from keras.models import load_model
model = load_model(sp_dir + "bondai_model_1.1.h5")


# In[178]:


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
    


# In[179]:


test_df


# In[180]:


df3


# In[ ]:


df3.to_csv(model_dir + "pred_data.csv")

