
# coding: utf-8

# In[43]:


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
root_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/"
data_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/raw/"
prep_dir = "/content/vdrive/My Drive/Colab Notebooks/Projects/Bondai/SP 500/data/prep/"


# In[ ]:


# Loading the csv tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header=None, names=["Tickers"])


# In[ ]:


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


# In[ ]:


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


# In[48]:


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


# ### References
# 
# 1. http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.insert.html
