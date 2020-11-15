#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Importing all the libriaries
import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# In[50]:


class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    # linear Regression Model: Fitting by Linear Algebra Approach
    def fit(self, X, y):
        n, m = X.shape
        
        X_train = X
        
        # Adding intecept column to the dataset matrix if intercept exists
        if self.fit_intercept:
            X_train = np.hstack((np.ones((n, 1)), X))
        
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y
        
        return self
    
    
    def predict(self, X):
        n, m = X.shape
        
        # Adding intecept column to the dataset matrix if intercept exists
        if self.fit_intercept:
            X_test = np.hstack((np.ones((n, 1)),X))
            
        y_pred = X_test @ self.w
        
        return y_pred
    
    
    def get_weights (self):
        return self.w
        


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


def linear_expression(x):
    return 7 + 5 * x   


# In[53]:


objects_num = 50
X = np.linspace(-5, 5, objects_num)
y = linear_expression(X) + np.random.randn(objects_num) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5)


# In[54]:


plt.figure(figsize=(10, 7))
plt.plot(X, linear_expression(X), label = 'real', c='g')
plt.scatter(X_train, y_train, label = 'train', c = 'b')
plt.scatter(X_test, y_test, label = 'test', c = 'orange')

plt.title("Generated dataset")
plt.grid(alpha=0.2)
plt.legend()
plt.show()


# In[55]:


regressor = MyLinearRegression()


regressor.fit(X_train[:, np.newaxis], y_train)

predictions = regressor.predict(X_test[:, np.newaxis])
w = regressor.get_weights()
w


# In[56]:


# Plotting chatter charts of train, test and tran_test
plt.figure(figsize=(18, 7))

ax = None

for i, types in enumerate ([['train','test'],['train'], ['test']]):
    ax = plt.subplot(1, 3, i + 1, sharey = ax)
    if 'train' in types:
        plt.scatter(X_train, y_train, label = 'train', c = 'b')
    if 'test' in types:
        plt.scatter(X_test, y_test, label = 'test', c = 'orange')
        
    plt.plot(X, linear_expression(X), label='real', c = 'g')
    plt.plot(X, regressor.predict(X[:,np.newaxis]), label = 'predicted', c = 'r')
    
    plt.ylabel('target')
    plt.xlabel('feature')
    plt.title(" ".join(types))
    plt.grid(alpha = 0.2)
    plt.legend()
    
plt.show()


# In[57]:


# SkLearn realisation
sk_reg = LinearRegression().fit(X_train[:,np.newaxis], y_train)

plt.figure(figsize = (10, 7))
plt.plot(X, linear_expression(X), label = 'real', c = 'g')

plt.scatter(X_train, y_train, label = 'train')
plt.scatter(X_test, y_test, label = 'test')
plt.plot(X, sk_reg.predict(X[:, np.newaxis]), label = 'sklearn', c = 'cyan', linestyle = ':')
plt.plot(X, regressor.predict(X[:, np.newaxis]), label = 'ours', c = 'r', linestyle = ':') 


plt.ylabel('target')
plt.xlabel('feature')
plt.title('Different Prediction')
plt.legend()
plt.show()


# In[58]:


from sklearn.metrics import mean_squared_error

train_predictions = regressor.predict(X_train[:, np.newaxis])
test_predictions = regressor.predict(X_test[:, np.newaxis])

print('Train MSE: ', mean_squared_error(y_train, train_predictions))
print('Test MSE: ', mean_squared_error(y_test, test_predictions))


# In[77]:


class MyGradientLinearRegression(MyLinearRegression):
    def __init__ (self, **kwargs):
        super().__init__(**kwargs) # passing parameters to a parent constructor
        self.w = None
        
    def fit(self, X, y, lr = 0.01, max_iter = 100):
        # Here we override the parent method
        # We should never forhet about intercept
        # lr stands for Learning Rate
        
        n, k = X.shape
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        self.losses = []
        
        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)
            
        for iter_num in range (max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))
            
            grad = self._calc_gradient(X_train, y, y_pred)
            
            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= grad*lr
       
        return self
    
    
    # Gardient Calculation
    def _calc_gradient(self, X, y, y_pred):
        n, k = X.shape
        grad = 2 * (y_pred - y)[:,np.newaxis] * X
        
        return grad.mean(axis=0)
    
    
    def get_losses(self):
        return self.losses
    


# In[78]:


regressor_1 = MyGradientLinearRegression(fit_intercept=True)
new_model = regressor_1.fit(X_train[:,np.newaxis], y_train, max_iter = 100).get_losses()

prediction = regressor_1.predict(X_test[:,np.newaxis])
w = regressor_1.get_weights()


# In[ ]:




