import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet

# Create sample data
data = {
    'Month': pd.date_range(start='2021-01', periods=24, freq='M'),
    'Sales': [100, 120, 130, 140, 160, 150, 180, 190, 210, 200, 230, 220,
              250, 240, 270, 260, 280, 270, 300, 310, 330, 320, 340, 330]
}

df = pd.DataFrame(data)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Data')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Prepare the data for modeling
X = (df['Month'] - df['Month'].min()) / pd.Timedelta(days=1)
X = X.values.reshape(-1, 1)

y = df['Sales'].values

# Build and fit the ARIMA model
order = (1, 1, 1)  # (p, d, q)
model_arima = sm.tsa.ARIMA(y, order=order)
results_arima = model_arima.fit()

# Forecast future values
forecast_steps = 12  # Forecasting 12 months ahead
forecast_arima = results_arima.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df['Month'].max(
), periods=forecast_steps+1, freq='M')[1:]  # Get future dates

# Create a new DataFrame for ARIMA forecasted values
forecast_df_arima = pd.DataFrame(
    {'Month': forecast_index, 'Forecast_Sales': forecast_arima})

# Prepare the data for Prophet modeling
df_prophet = df.rename(columns={'Month': 'ds', 'Sales': 'y'})

# Create and fit the Prophet model
model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Make future predictions with Prophet
future_months = pd.date_range(start='2023-08', periods=12, freq='M')
future_df = pd.DataFrame({'ds': future_months})
forecast_prophet = model_prophet.predict(future_df)
forecast_df_prophet = forecast_prophet[['ds', 'yhat']].rename(
    columns={'yhat': 'Forecast_Sales'})

# Visualize the forecasts
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o', label='Actual Sales')
plt.plot(forecast_df_arima['Month'], forecast_df_arima['Forecast_Sales'],
         marker='o', label='Forecasted Sales (ARIMA)')
plt.plot(forecast_df_prophet['ds'], forecast_df_prophet['Forecast_Sales'],
         marker='o', label='Forecasted Sales (Prophet)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Data and Forecasts (ARIMA and Prophet)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
