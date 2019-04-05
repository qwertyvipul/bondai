
# coding: utf-8

# In[192]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')
get_ipython().system('pip install pandas-datareader')
from pandas_datareader import data as pdr
get_ipython().system('pip install fix-yahoo-finance')
import fix_yahoo_finance as yf


# In[193]:



share = "GOOGL"
dataset = pdr.get_data_yahoo(share)
dataset = dataset.dropna() # removing missing value rows
dataset


# In[ ]:


openi=list(dataset['Open'])


# In[ ]:


close=list(dataset['Close'])


# In[196]:


close


# In[197]:


openi


# In[ ]:


change=[]
for i,j in zip(openi,close):
  change.append(j-i)


# In[ ]:


date=(dataset.index.tolist())



# In[200]:


plt.plot(date, openi, '--',linewidth=4.0)
plt.plot(date, close, '--',linewidth=4.0)
plt.axis([date[0], date[-1],int(min(min(openi),min(close))//2),int(max(openi)+max(close))])

plt.legend(["Opening Price","Closing Price"])
plt.title("Google LLC Stock Prices")
plt.xlabel("Dates")
plt.ylabel("Stock Price")
plt.show()


# In[201]:


plt.plot(date[-300:], change[-300:], '--',linewidth=1.0)
# plt.plot(date[1022:], close[1022:], '--')
plt.axis([date[-300], date[-1],int(min(change)*1.5),int(max(change)+max(change))])

l=plt.legend(["Change between opening and closing price"])
plt.title("Google LLC Stock Prices")
#plt.xticks(rotation=90)
plt.xlabel("Dates")
plt.ylabel("Change between opening and closing price")
plt.show()


# In[202]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups=len(date[-100:])
op = openi[-100:]
cl=close[-100:]
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.07
opacity = 0.8
 
rects1 = plt.bar(index, op, bar_width,
alpha=opacity,
color='b',
label='Open Price')
 
rects2 = plt.bar(index + bar_width, cl, bar_width,
alpha=opacity,
color='r',
label='Close Price')
 
plt.xlabel('Dates')
plt.ylabel('Stock Price')
plt.title('Goolgle LLC Stock Prices')
plt.xticks(rotation=90)
plt.xticks(index + bar_width, date[-100:])
plt.legend()
 
plt.tight_layout()
plt.show()


# In[203]:



share = "RCOM.NS"
dataset = pdr.get_data_yahoo(share)
dataset = dataset.dropna() # removing missing value rows
dataset


# In[204]:


openi=list(dataset['Open'])
close=list(dataset['Close'])
date=(dataset.index.tolist())
date


# In[205]:


print(str(date[0]))
c=0
for i in range(len(date)):
  if(str(date[i])==('2014-02-17 00:00:00')):
    c=i
    break
c    
  
  


# In[206]:


plt.plot(date[:], openi[:], '--')
plt.plot(date[:], close[:], '--')
plt.axis([date[0], date[-1],int(min(min(openi),min(close))//2),int(max(openi)+max(close))])

l=plt.legend(["Opening Price","Closing Price"])
plt.title("Reliance Comm Stock Prices")
#plt.xticks(rotation=90)
plt.xlabel("Dates")
plt.ylabel("Stock Price")
plt.show()


# In[ ]:


change=[]
for i,j in zip(openi,close):
  change.append(j-i)


# In[208]:


change


# In[ ]:


get_ipython().magic('matplotlib inline')

get_ipython().magic("config InlineBackend.print_figure_kwargs = {'bbox_inches':None}")
plt.style.use('fivethirtyeight')
get_ipython().magic("config InlineBackend.rc = {'font.size': 10, 'figure.figsize': (16, 8.0), 'figure.facecolor': (1, 1, 1, 0), 'figure.subplot.bottom': 0.125, 'figure.edgecolor': (1, 1, 1, 0), 'figure.dpi': 72}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[210]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups=len(date[-100:])
op = openi[-100:]
cl=close[-100:]
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.07
opacity = 0.8
 
rects1 = plt.bar(index, op, bar_width,
alpha=opacity,
color='b',
label='Open Price')
 
rects2 = plt.bar(index + bar_width, cl, bar_width,
alpha=opacity,
color='r',
label='Close Price')
 
plt.xlabel('Dates')
plt.ylabel('Stock Price')
plt.title('Reliance Comm Stock Prices')
plt.xticks(rotation=90)
plt.xticks(index + bar_width, date[-100:])
plt.legend()
 
plt.tight_layout()
plt.show()


# In[211]:


plt.plot(date[1022:], change[1022:], '--',linewidth=1.0)
#plt.plot(date[1022:], close[1022:], '--')
plt.axis([date[1022], date[-1],int(min(change)*1.5),int(max(change)+max(change))])

l=plt.legend(["Change between opening and closing price"])
plt.title("Reliance Comm Stock Prices")
#plt.xticks(rotation=90)
plt.xlabel("Dates")
plt.ylabel("Change between opening and closing price")
plt.show()

