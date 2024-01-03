import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Load the dataset
file_path = '/Users/paramanandbhat/Downloads/energy consumption 2.csv'
data = pd.read_csv(file_path)

# Convert 'DATE' column to datetime
data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%Y')

# Splitting the dataset into training and test data
# Typically, a common split ratio is 80% for training and 20% for testing.
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

train_data = train_data.copy()

print(train_data.head(), test_data.head())

# Required Preprocessing 
train_data['timestamp'] = pd.to_datetime(train_data['DATE'], format='%Y-%m-%d')
train_data.index = train_data.timestamp

test_data['timestamp'] = pd.to_datetime(test_data['DATE'], format='%Y-%m-%d')
test_data.index = test_data.timestamp

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['ENERGY_INDEX'], label='train_data')
plt.plot(test_data.index,test_data['ENERGY_INDEX'], label='valid')
plt.legend(loc='best')
plt.title("Train and Validation Data")
plt.show()

# Stationarity Test
# dickey fuller, KPSS
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(timeseries):
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(train_data['ENERGY_INDEX'])    

'''
Test Statistic : 1.589352
ritical Value (1%)             -3.438995
Critical Value (5%)             -2.865355
Critical Value (10%)            -2.568802

Test Statistic > Critical value
p-value                          0.997825
Value of p is also higher 
Series is non stationary.
'''
#Verify series is non stationary using KPSS test

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


kpss_test(train_data['ENERGY_INDEX'])

'''
If the test statistic is greater than the critical value, we reject the null hypothesis (series is not stationary). If the test statistic is less than the critical value, if fail to reject the null hypothesis (series is stationary).  **Here test statistic is > than critical. Hence series is not stationary**

Alternatively, we can use the p-value to make the inference. If p-value is less than 0.05, we can reject the null hypothesis. And say that the series is not stationary.

'''

# Making Series Stationary

train_data['ENERGY_INDEX_DIFF'] = train_data['ENERGY_INDEX'] - train_data['ENERGY_INDEX'].shift(1)

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['ENERGY_INDEX'], label='train_data')
plt.plot(train_data.index,train_data['ENERGY_INDEX_DIFF'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

train_data['ENERGY_INDEX_LOG'] = np.log(train_data['ENERGY_INDEX'])
train_data['ENERGY_INDEX_LOG_DIFF'] = train_data['ENERGY_INDEX_LOG'] - train_data['ENERGY_INDEX_LOG'].shift(1)

plt.figure(figsize=(12,8))
plt.plot(train_data.index,train_data['ENERGY_INDEX_LOG_DIFF'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

adf_test(train_data['ENERGY_INDEX_LOG_DIFF'].dropna())

kpss_test(train_data['ENERGY_INDEX_LOG_DIFF'].dropna())

# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train_data['ENERGY_INDEX_LOG_DIFF'].dropna(), lags=15)
plot_pacf(train_data['ENERGY_INDEX_LOG_DIFF'].dropna(), lags=15)
plt.show()

# Preprocessing for SARIMA model
train_data['DATE'] = pd.to_datetime(train_data['DATE'], format='%Y-%m-%d')
train_data.set_index('DATE', inplace=True)

from statsmodels.tsa.statespace.sarimax import SARIMAX
# SARIMA Model
# Preprocessing for SARIMA model on entire dataset
data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d')
data.set_index('DATE', inplace=True)

# Fit the model to the entire dataset
full_model = SARIMAX(data['ENERGY_INDEX'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
full_model_fit = full_model.fit(disp=False)

# Forecast 36 months ahead from the end of the dataset
last_date = data.index[-1]
forecast_full = full_model_fit.forecast(steps=36)

# Adding forecast dates to the forecast series
forecast_dates_full = pd.date_range(start=last_date, periods=36, freq='M') # Assuming monthly frequency
forecast_full.index = forecast_dates_full

# Plotting the original series and full forecast
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['ENERGY_INDEX'], label='Original Data')
plt.plot(forecast_full.index, forecast_full, label='Future Forecast', color='red')
plt.title('Energy Consumption Forecast on Full Dataset')
plt.xlabel('Date')
plt.ylabel('Energy Index')
plt.legend()
plt.show()