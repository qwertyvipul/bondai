
# coding: utf-8

# In[3]:


"""
No need to execute this block when working on local system.
"""

# Mount Google Drive
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


# Loading the csv tickers, train_tickers and test_tickers file
import pandas as pd
ticker_list_df = pd.read_csv(root_dir + "ticker_list.csv", header = None, names = ["Tickers"])
train_tickers_df = pd.read_csv(root_dir + "train_tickers.csv", header = None, names = ["Train Tickers"])
test_tickers_df = pd.read_csv(root_dir + "test_tickers.csv", header = None, names = ["Test Tickers"])


# In[ ]:


train_set = set(train_tickers_df["Train Tickers"].tolist())
test_set = set(test_tickers_df["Test Tickers"].tolist())


# In[ ]:


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


# In[ ]:


def read_data_from_file(ticker):
    df = pd.read_csv(prep_dir + ticker + ".csv")
    return df[["net_income", "op_income", "gross_profit", "crr_asst", "ncrr_asst", "crr_libt", "ncrr_libt"]].values


# In[9]:


X_train, y_train = list(), list()
X_test, y_test = list(), list()

n_steps = 2

counter = 0
for ticker in ticker_list_df["Tickers"]:
    counter += 1
    print("Fetching data for: " + ticker + "(" + str(counter) + "/"+ str(len(ticker_list_df)) + ")")
    data = read_data_from_file(ticker)
    in_out_split(data, n_steps)


# In[10]:


print(len(X_train), len(y_train))
print(len(X_test), len(y_test))


# In[ ]:


import numpy as np
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


# In[12]:


n_features = X_train.shape[2]
n_features


# In[13]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[14]:


model = Sequential()
model.add(LSTM(100, activation = "relu", return_sequences = True, input_shape = (n_steps, n_features)))
model.add(LSTM(100, activation = "relu"))
model.add(Dense(n_features))


# In[ ]:


model.compile(optimizer = "adam", loss = "mse")


# In[17]:


model.fit(X_train, y_train, epochs = 400, batch_size = 1, shuffle = True, verbose = 1)


# In[18]:


model.evaluate(X_test, y_test, verbose=1)


# In[19]:


X_test[1]


# In[22]:


test_input = X_test[1].reshape(1, n_steps, n_features)
model.predict(test_input, verbose = 1)


# In[21]:


y_test[1]


# In[ ]:


model.save(root_dir + "bondai_model_1.0.h5")

