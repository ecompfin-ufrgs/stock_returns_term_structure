#!/usr/bin/env python
# coding: utf-8

# In[24]:


import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[6]:


start_date = '2020-01-01'
end_date = '2024-01-01'


market = yf.download('^BVSP', start=start_date, end=end_date)
petr4 = yf.download('PETR4.SA', start=start_date, end=end_date)


# In[7]:


cdi_data = pd.read_csv('Downloads/selic.csv',sep=';', parse_dates=['Date'],dayfirst=True)
cdi_data.set_index('Date', inplace=True)
cdi_data.index = pd.to_datetime(cdi_data.index, dayfirst=True)
cdi_data['return'] = cdi_data['return'].str.replace(',', '.').astype(float)
cdi_data.rename(columns={'return': 'CDI_Return'}, inplace=True)


# In[8]:


ret_petr4 = petr4['Adj Close'].pct_change().dropna()
ret_market = market['Adj Close'].pct_change().dropna()


# In[9]:


data = pd.concat([ret_petr4, ret_market, cdi_data], axis=1).dropna()
data.columns = ['Petr4_Return', 'Market_Return', 'CDI_Return']


# In[10]:


data['Petr4_Excess'] = data['Petr4_Return'] - data['CDI_Return']
data['Market_Excess'] = data['Market_Return'] - data['CDI_Return']


# In[11]:


data['Petr4_Excess'] = data['Petr4_Excess'] * 100
data['Market_Excess'] = data['Market_Excess'] * 100


# In[12]:


am = arch_model(data['Petr4_Excess'], x=data[['Market_Excess']], mean='ARX', lags=0,
                vol='GARCH', p=1, q=1, dist='normal')


# In[13]:


res = am.fit()


# In[14]:


am_market = arch_model(data['Market_Excess'], mean='Constant', vol='GARCH', p=1, q=1)
res_market = am_market.fit(disp='off')


# In[15]:


sigma_market = res_market.conditional_volatility


# In[16]:


lambda_market = data['Market_Excess'].mean() / sigma_market.mean()


# In[17]:


prm_market = lambda_market * sigma_market


# In[18]:


beta = res.params['Market_Excess']


# In[19]:


prm_petr4 = beta * prm_market


# In[20]:


petr4_expected_return = data['CDI_Return'] + prm_petr4


# In[21]:


data['Prêmio_Risco_Mercado'] = prm_market
data['Prêmio_Risco_Ativo'] = prm_petr4
data['petr4_expected_return'] = petr4_expected_return


# In[25]:


plt.figure(figsize=(12,6))
plt.plot(data.index, data['Prêmio_Risco_Ativo'], label='Prêmio de Risco do Ativo')
plt.title('Prêmio de Risco diário de PETR4 ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Prêmio de Risco (%)')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.show()


# In[ ]:




