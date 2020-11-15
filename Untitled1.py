#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BitMex API
# ID: s4EYnAN6Plp1da_SfhPf5UYC 
# Key: UZDtiUQPpNcLbJ6P-AdIRvl1XWg9RXcc0gpwgz2C36Jy2867


# In[98]:


import bitmex
import requests, json
import time as time
import pandas as pd


bitmex_api_key = "s4EYnAN6Plp1da_SfhPf5UYC "
bitmex_api_secret = "UZDtiUQPpNcLbJ6P-AdIRvl1XWg9RXcc0gpwgz2C36Jy2867"

client = bitmex.bitmex(test = False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)


# In[99]:


def get_market_data():
    binSize='1m' # You can change the bin size as needed
    symbol = 'XBTUSD'

    past_minute_data = client.Trade.Trade_getBucketed(binSize=binSize, count=10, symbol=symbol, reverse=True).result()[0][0]

    processed_min_data = {}
    timestamp_minute = str(past_minute_data["timestamp"]).split(':')[0] + ":" +                        str(past_minute_data["timestamp"]).split(':')[1] + ":00"

    processed_min_data["Underlying"] = past_minute_data["symbol"]
    processed_min_data["Date"] = timestamp_minute
    processed_min_data["Open"] = past_minute_data["open"]
    processed_min_data["Close"] = past_minute_data["close"]
    processed_min_data["Volume"] = past_minute_data["volume"]
    processed_min_data["High"] = past_minute_data["high"]
    processed_min_data["Low"] = past_minute_data["low"]
    
    return processed_min_data


# In[139]:


column_names = ['Underlying',
                'Date',
                'Open',
                'Close',
                'Volume',
                'High',
                'Low']

df = pd.DataFrame(columns = column_names)

max_duration = 3600
keep_recording = True
time_step = 60
duration = time_step

while keep_recording:
    data = get_market_data()
    df = df.append(data, ignore_index=True)
    
    time.sleep(time_step)
    duration += time_step
    
    if duration > max_duration:
   #     if duration % 4 != 0:
    #        df.to_csv('Bitmex_1m_data.csv', mode='a', header=False)
        keep_recording = False
        
  #  if duration % 4 == 0:
   #     df.to_csv('Bitmex_1m_data.csv', mode='a', header=False)
    #    df.drop(df.index, inplace=True)
    
    

df.tail()

    


# In[ ]:


df.to_csv('Bitmex_1m_data.csv', index = False)


# In[ ]:


df.to_csv('Bitmex_1m_data.csv', mode='a', header=True)


# In[ ]:


kp = pd.read_csv('Bitmex_1m_data.csv')


# In[134]:


kp


# In[ ]:


df


# In[137]:


df.drop(df.index, inplace=True)


# In[138]:


df


# In[ ]:




