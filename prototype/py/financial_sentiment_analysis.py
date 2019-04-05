
# coding: utf-8

# In[1]:



import nltk
import warnings
warnings.filterwarnings('ignore')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


# In[2]:



from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pprint

date_sentiments = {}

for i in range(1,11):
    page = urlopen('https://www.businesstimes.com.sg/search/google?page='+str(i)).read()#company whose data to be fetched
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", {"class": "media-body"})
    for post in posts:
        time.sleep(1)
        url = post.a['href']
        date = post.time.text
        print(date, url)
        try:
            
            link_page = urlopen(url).read()
            
        except:
            break
            
#           url = url[:-2]  
#             print(url)
            link_page = urlopen(url).read()
        link_soup = BeautifulSoup(link_page)
        sentences = link_soup.findAll("p")
        passage = ""
        for sentence in sentences:
            passage += sentence.text
        sentiment = sia.polarity_scores(passage)['compound']
        date_sentiments.setdefault(date, []).append(sentiment)

date_sentiment = {}

for k,v in date_sentiments.items():
    date_sentiment[datetime.strptime(k, '%d %b %Y').date() + timedelta(days=1)] = round(sum(v)/float(len(v)),3)

earliest_date = min(date_sentiment.keys())

print(date_sentiment)


# In[ ]:


get_ipython().magic('matplotlib inline')

get_ipython().magic("config InlineBackend.print_figure_kwargs = {'bbox_inches':None}")
plt.style.use('fivethirtyeight')
get_ipython().magic("config InlineBackend.rc = {'font.size': 5, 'figure.figsize': (16.0, 8.0), 'figure.facecolor': (1, 1, 1, 0), 'figure.subplot.bottom': 0.125, 'figure.edgecolor': (1, 1, 1, 0), 'figure.dpi': 72}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


print(type(date_sentiment))


# In[ ]:


Date=list(date_sentiment.keys())
Sentiment_values=list(date_sentiment.values())


# In[ ]:


print(Date[:10],Sentiment_values[:10])


# In[ ]:



plt.plot(Date, Sentiment_values, '--')
plt.axis([Date[0], Date[-1],-1,1])


plt.title("Google LLC NewsPaper Based Finanacial Sentiment Analysis")
plt.xlabel("Dates")
plt.ylabel("Sentiment Score(Postive(>0),Negative(<0))")
plt.show()


# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pprint

date_sentiments = {}

for i in range(1,11):
    page = urlopen('https://www.businesstimes.com.sg/search/reliance?page='+str(i)).read()#company whose data to be fetched
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", {"class": "media-body"})
    for post in posts:
        time.sleep(1)
        url = post.a['href']
        date = post.time.text
        print(date, url)
        try:
            
            link_page = urlopen(url).read()
            
        except:
            
#           url = url[:-2]  
#             print(url)
            link_page = urlopen(url).read()
        link_soup = BeautifulSoup(link_page)
        sentences = link_soup.findAll("p")
        passage = ""
        for sentence in sentences:
            passage += sentence.text
        sentiment = sia.polarity_scores(passage)['compound']
        date_sentiments.setdefault(date, []).append(sentiment)

date_sentiment = {}

for k,v in date_sentiments.items():
    date_sentiment[datetime.strptime(k, '%d %b %Y').date() + timedelta(days=1)] = round(sum(v)/float(len(v)),3)

earliest_date = min(date_sentiment.keys())

print(date_sentiment)


# In[ ]:


Date=list(date_sentiment.keys())
Sentiment_values=list(date_sentiment.values())


# In[ ]:





# In[ ]:


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.plot(Date, Sentiment_values, '--')
plt.axis([Date[0], Date[-1],-1,1])


plt.title("Reliance Comm NewsPaper Based Finanacial Sentiment Analysis")
plt.text(2, 0.65,"If sentiment increasing it is good to invest ", fontdict=font)
plt.xlabel("Dates")
plt.ylabel("Sentiment Score(Postive(>0),Negative(<0))")
plt.show()


# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().system('pip install pandas-datareader')
from pandas_datareader import data as pdr
get_ipython().system('pip install fix-yahoo-finance')
import fix_yahoo_finance as yf


# In[ ]:



share = "GOOGL"
dataset = pdr.get_data_yahoo(share)
dataset = dataset.dropna() # removing missing value rows
dataset


# In[ ]:


openi=list(dataset['Open'])
close=list(dataset['Close'])
date=(dataset.index.tolist())


# In[ ]:


plt.plot(date, openi, '--')
plt.plot(date, close, '--')
plt.axis([date[0], date[-1],int(min(min(openi),min(close))//2),int(max(openi)+max(close))])

plt.legend(["Opening Price","Closing Price"])
plt.title("Google LLC Stock Prices")
plt.xlabel("Dates")
plt.ylabel("Stock Price")
plt.show()


# In[ ]:


share = "RCOM.NS"
dataset = pdr.get_data_yahoo(share)
dataset = dataset.dropna() # removing missing value rows
dataset


# In[ ]:


openi=list(dataset['Open'])
close=list(dataset['Close'])
date=(dataset.index.tolist())


# In[ ]:


plt.plot(date, openi, '--')
plt.plot(date, close, '--')
plt.axis([date[0], date[-1],int(min(min(openi),min(close))//2),int(max(openi)+max(close))])

plt.legend(["Opening Price","Closing Price"])
plt.title("Reliance Comm Stock Prices")
plt.xlabel("Dates")
plt.ylabel("Stock Price")
plt.show()

