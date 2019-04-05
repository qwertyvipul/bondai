
# coding: utf-8

# In[1]:


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


# Loading the csv files
import pandas as pd
actual_df = pd.read_csv(model_dir + "actual_data.csv")
pred_df = pd.read_csv(model_dir + "pred_data.csv")


# In[ ]:


actual_df["act_short_term"] = actual_df["crr_asst"]/actual_df["crr_libt"]
actual_df["act_long_term"] = actual_df["ncrr_asst"]/actual_df["ncrr_libt"]
actual_df["act_overall"] = (actual_df["crr_asst"] + actual_df["ncrr_asst"])/(actual_df["crr_libt"] + actual_df["ncrr_libt"])


# In[ ]:


actual_df


# In[ ]:


pred_df["pred_short_term"] = pred_df["crr_asst"]/pred_df["crr_libt"]
pred_df["pred_long_term"] = pred_df["ncrr_asst"]/pred_df["ncrr_libt"]
pred_df["pred_overall"] = (pred_df["crr_asst"] + pred_df["ncrr_asst"])/(pred_df["crr_libt"] + pred_df["ncrr_libt"])


# In[ ]:


pred_df


# In[ ]:


actual_rating = actual_df[["Ticker", "act_short_term", "act_long_term", "act_overall"]]
actual_rating.set_index("Ticker", inplace = True)


# In[ ]:


actual_rating


# In[ ]:


pred_rating = pred_df[["Ticker", "pred_short_term", "pred_long_term", "pred_overall"]]
pred_rating.set_index("Ticker", inplace = True)
pred_rating


# In[ ]:


rating_df = pd.concat([actual_rating, pred_rating], axis = 1)
rating_df


# In[ ]:


rounded_df = round(rating_df)
rounded_df


# In[ ]:


# rounded_df.to_csv(model_dir + "credit_ratings_1.0.csv")
rounded_df = pd.read_csv(model_dir + "credit_ratings_1.0.csv")


# In[34]:


import matplotlib.pyplot as plt
plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_short_term"], "b", label = "actual")
plt.plot(rounded_df["pred_short_term"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Short Term Credit Rating")
plt.show()


# In[35]:


import matplotlib.pyplot as plt
plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_short_term"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_short_term"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Short Term Credit Rating")
plt.show()


# In[36]:


plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_long_term"], "b", label = "actual")
plt.plot(rounded_df["pred_long_term"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Long Term Credit Rating")
plt.show()


# In[37]:


plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_long_term"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_long_term"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Long Term Credit Rating")
plt.show()


# In[38]:


plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_overall"], "b", label = "actual")
plt.plot(rounded_df["pred_overall"], "r", label = "predicted")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Overall Credit Rating")
plt.show()


# In[39]:


plt.figure(figsize = (15, 6))
plt.plot(rounded_df["act_overall"], "b", label = "actual", marker = "o", linestyle = "none")
plt.plot(rounded_df["pred_overall"], "r", label = "predicted", marker = "o", linestyle = "none")
plt.legend(loc='upper right')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90, fontsize = 8)
plt.xlabel("Overall Credit Rating")
plt.show()


# In[ ]:


import numpy as np
plt.hist([rounded_df["act_short_term"], rounded_df["pred_short_term"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Short Term Credit Rating")
plt.show()


# In[ ]:


plt.hist([rounded_df["act_long_term"], rounded_df["pred_long_term"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Long Term Credit Rating")
plt.show()


# In[ ]:


plt.hist([rounded_df["act_overall"], rounded_df["pred_overall"]], label=['actual', 'predicted'], color = ["b", "r"])
plt.legend(loc='upper right')
plt.xlabel("Overall Credit Rating")
plt.show()

