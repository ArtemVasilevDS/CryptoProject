#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import *

K = 8000
S = np.linspace(7000, 9000, 100)
h = np.maximum(S-K,0)

plt.figure()
plt.plot(S, h, lw=2.5)
plt.ylabel('Inner value of european call option')
plt.xlabel('Index Level $S_T$ at Maturity')
plt.title('Payout Plot')
plt.grid(True, lw=0.5, ls= ':')
plt.show()


# In[55]:


class Option:
    def __init__(self, spot, strike, t_0, T, rate, vol):
        self.spot = spot
        self.strike = strike
        self.t_0 = t_0
        self.T = T
        self.rate = rate
        self.vol = vol
        
    def reset_param(self, spot, strike, t_0, T, rate, vol):
        self.spot = spot
        self.strike = strike
        self.t_0 = t_0
        self.T = T
        self.rate = rate
        self.vol = vol
    
    
    def _N_func(self, x):
        return norm.cdf(x)
    
    
    def d_1(self):
        # Let's rewrite variables for return funciton to look more readable
        spot, strike, rate, vol, T, t_0 = self.spot, self.strike, self.rate, self.vol, self.T, self.t_0
        
        return (log(spot/strike) + (rate + (vol ** 2.0)/2.0)*T)/(vol * sqrt(T-t_0))
    
    
    def d_2(self):
        # Let's rewrite variables for return funciton to look more readable
        vol, T, t_0 = self.vol, self.T, self.t_0
        
        return self.d_1() - vol * sqrt(T-t_0)
    
    
    def BSM_price(self):
        # Let's rewrite variables for return funciton to look more readable
        spot, strike, rate, T, t_0 = self.spot, self.strike, self.rate, self.T, self.t_0
        
        return self._N_func(self.d_1())*spot - strike*e(-rate*(T-t_0))*self._N_func(self.d_2())


# In[56]:


K = 8000
S = np.linspace(7000, 9000, 100)
h = np.maximum(S-K,0)

r = 0.025
vol = 0.2
t_0 = 0.0
T = 1.0
S_0 =8000
Option(S_0, K, t_0, T, r, vol).BSM_price()
#bsm = [Option(S_0, K, t_0, T, r, vol).BSM_price() for S_0 in S] 

plt.figure()
plt.plot(S, h, lw=2.5)
plt.ylabel('Inner value of european call option')
plt.xlabel('Index Level $S_T$ at Maturity')
plt.title('Payout Plot')
plt.grid(True, lw=0.5, ls= ':')
plt.show()


# In[32]:


log(e)


# In[34]:


sqrt(4)


# In[ ]:




