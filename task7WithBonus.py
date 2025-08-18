import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
print("Task 7: Walmart Sales Forecasting")

# Load data
stores=pd.read_csv('Datasets/Walmart_Sales_Forecast/stores.csv')
train=pd.read_csv('Datasets/Walmart_Sales_Forecast/train.csv')
train['IsHoliday'] = train['IsHoliday'].map({True: 1, False: 0}).astype(int)
df = pd.read_csv('Datasets/Walmart_Sales_Forecast/features.csv')
print("Data loaded successfully.")

# Convert 'Date' to datetime
train['Date']=pd.to_datetime(train['Date'])
df['Date']=pd.to_datetime(df['Date'])

df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis =1 , inplace = True )
df = df.merge(stores, on='Store', how='left')
df['IsHoliday'] = df['IsHoliday'].map({True: 1, False: 0}).astype(int)
df['Type'] = df['Type'].map({'A': 1, 'B': 2,'C' :3}).astype(int)

train_merged = train.merge(df, on=['Store', 'Date', 'IsHoliday'], how='left')
print("Data merged successfully.")

# Sort and basic prep
train_merged = train_merged.sort_values('Date')
train_merged = train_merged.set_index('Date')

# Feature engineering
train_merged['day'] = train_merged.index.day
train_merged['month'] = train_merged.index.month
train_merged['year'] = train_merged.index.year
train_merged['dayofweek'] = train_merged.index.dayofweek
train_merged['lag_1'] = train_merged['Weekly_Sales'].shift(1)
train_merged['lag_7'] = train_merged['Weekly_Sales'].shift(7)
train_merged['rolling_mean_7'] = train_merged['Weekly_Sales'].rolling(window=7).mean()
print("Features engineered successfully.")

#Assign X and y
features = ['day', 'month', 'year', 'dayofweek', 'lag_1', 'lag_7', 'rolling_mean_7', 'IsHoliday']
train_ready = train_merged.dropna(subset=['lag_1', 'lag_7', 'rolling_mean_7'])

# Use the last N weeks of train_merged as validation
validation_size = int(0.2 * len(train_ready))
train_part = train_ready.iloc[:-validation_size]
val_part = train_ready.iloc[-validation_size:]

X_train, y_train = train_part[features], train_part['Weekly_Sales']
X_val, y_val = val_part[features], val_part['Weekly_Sales']
print("Data split into training and validation sets.")

# Train model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

# Plot actual vs predicted for validation
plt.figure(figsize=(12,6))
plt.plot(val_part.index, y_val, label='Actual')
plt.plot(val_part.index, y_pred_rf, label='Predicted (RF)')
plt.legend()
plt.title('Actual vs Predicted Sales (Random Forest, Validation)')
plt.show()

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_rf)))

# Rolling averages and seasonal decomposition
train_merged['rolling_mean_7'] = train_merged.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(7).mean())
plt.figure(figsize=(12,6))
plt.plot(train_merged['Weekly_Sales'], label='Weekly_Sales')
plt.plot(train_merged['rolling_mean_7'], label='7-day Rolling Mean')
plt.legend()
plt.title('Sales and Rolling Average')
plt.show()

result = seasonal_decompose(train_merged['Weekly_Sales'], model='additive', period=30)
result.plot()
plt.show()

# XGBoost with time-aware validation
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_val)

plt.figure(figsize=(12,6))
plt.plot(val_part.index, y_val, label='Actual')
plt.plot(val_part.index, y_pred_xgb, label='Predicted (XGBoost)')
plt.legend()
plt.title('Actual vs Predicted Sales (XGBoost)')
plt.show()

print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_xgb)))