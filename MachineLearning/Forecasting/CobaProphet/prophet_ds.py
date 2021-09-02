

# instalasi
 !pip install pystan
 !pip install fbprophet

import warnings; 
warnings.simplefilter('ignore')
import pandas as pd
from fbprophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/master/examples/example_air_passengers.csv') #percobaan 1 data penerbangan
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv') #percobaan 1 data retail

df.head(10)

df.info()

plt.plot(df['ds'],df['y'])
plt.show()

"""latih data"""

m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)

# # Python
# future = m.make_future_dataframe(periods=7)
# future.tail()

"""Forecasting"""

future = m.make_future_dataframe(periods=365*6,freq='D')
forecast = m.predict(future)
forecast.head()

plot1 = m.plot(forecast)

plt2 = m.plot_components(forecast)