import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# feature eng

# for log changes
epsilon = 1e-1000000

# Distance from the moving averages
for m in [10, 20, 30, 50, 100]:
    data[f'feat_dist_from_ma_{m}'] = data['Close']/data['Close'].rolling(m).mean()-1
    
# Distance from n day max/min
for m in [3, 5, 10, 15, 20, 30, 50, 100]:
    data[f'feat_dist_from_max_{m}'] = data['Close']/data['High'].rolling(m).max()-1
    data[f'feat_dist_from_min_{m}'] = data['Close']/data['Low'].rolling(m).min()-1
    
# Price distance
for m in [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]:
    data[f'feat_price_dist_{m}'] = data['Close']/data['Close'].shift(m)-1

for lag in range(1, 4):
    data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
    data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)

data['VolumeImbalance'] = data['BaseAssetVolume'] - data['TakerBuyBaseAssetVolume']
data['VolumeRatio'] = data['TakerBuyVolume'] / data['Volume']
data['VolumeChange'] = data['Volume'].diff()

data['TimeDeltaSeconds'] = (data['CloseTime'] - data['OpenTime']).dt.total_seconds()
data['TradeFrequency'] = data['NumberOfTrades'] / data['TimeDeltaSeconds']
data['TradeSize'] = data['BaseAssetVolume'] / data['NumberOfTrades']
data['TradeSizeChange'] = data['TradeSize'].diff()

data['AvgVolumePerTrade'] = data['BaseAssetVolume'] / data['NumberOfTrades']
data['AvgTradesPerVolume'] = data['NumberOfTrades'] / data['BaseAssetVolume']
data['LiquidityScore'] = data['AvgVolumePerTrade'] * data['AvgTradesPerVolume']

data['BuyToSellRatio'] = data['TakerBuyBaseAssetVolume'] / data['BaseAssetVolume']
data['TakerBuyPercentage'] = data['TakerBuyVolume'] / data['Volume']
data['SentimentScore'] = data['BuyToSellRatio'] * data['TakerBuyPercentage']

data['Volatility'] = (data['High'] - data['Low']) / data['Open']
data['VolatilityChange'] = data['Volatility'].diff()

data['PriceChange'] = data['Close'] - data['Open']
data['VolumeWeightedPriceChange'] = data['PriceChange'] * data['Volume']
data['MomentumScore'] = data['PriceChange'] * data['VolumeChange']

data['RollingMeanVolume'] = data['Volume'].rolling(window=10).mean()
data['RollingStdVolume'] = data['Volume'].rolling(window=10).std()
data['RollingMeanPrice'] = data['Close'].rolling(window=10).mean()
data['RollingStdPrice'] = data['Close'].rolling(window=10).std()

data['EMA_Volume'] = data['Volume'].ewm(span=10, adjust=False).mean()
data['EMA_Price'] = data['Close'].ewm(span=10, adjust=False).mean()

data['DayOfWeek'] = data['OpenTime'].dt.dayofweek
data['HourOfDay'] = data['OpenTime'].dt.hour
# data['TradingSession'] = pd.cut(data['HourOfDay'], bins=[0, 8, 12, 13, 17, 24], labels=['Night', 'Morning', 'Lunch', 'Afternoon', 'Evening'], right=False)
data['ZScorePrice'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
data['ZScoreVolume'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()

data['high_low_range'] = data['High'] / data['Low']
data['log_volume_change'] = np.log(data['Volume'] / data['Volume'].shift(1) + epsilon)
data['log_trade_count_change'] = np.log(data['NumberOfTrades'] / data['NumberOfTrades'].shift(1) + epsilon)
data['obv'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
data['price_volume_ratio'] = data['Close'] / data['Volume']

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.ffill()
data.dropna(inplace=True)

periods_ahead = 5
data['time_t'] = np.where(data['Close'].pct_change(periods=periods_ahead).shift(-periods_ahead) > 0, 1, 0)

# Add more time series features 
data['time_t+1'] = data.time_t.shift(1)
data['time_t+2'] = data.time_t.shift(2)
data = data.dropna()

total_columns = data.shape[1]

X = data.iloc[:,11:total_columns - 1]  # achieve higher accuracy, lower loss
y = data.iloc[:,total_columns - 1]   

split = int(len(data) * 0.5)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

test_start_date = data['OpenTime'].iloc[split].date().strftime('%Y-%m-%d')
print(f"Test set starts on: {test_start_date}")

# normalize the data 
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
# y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1,1))
X_test= x_scaler.transform(X_test)
# X_test= y_scaler.transform(np.array(y_test).reshape(-1,1))

# Combine X_train_scaled and X_test_scaled for VIF calculation
X_scaled_combined = np.vstack((X_train, X_test))

# Calculate VIF for all features
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled_combined, i) for i in range(X.shape[1])]

    # Drop features with VIF > 10
    high_vif_features = vif_data[vif_data['VIF'] > 10]['feature']
    X_reduced = X.drop(columns=high_vif_features)

    # Recalculate VIF for the reduced feature set
    vif_data_reduced = pd.DataFrame()
    vif_data_reduced["feature"] = X_reduced.columns
    vif_data_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(len(X_reduced.columns))]

    high_vif_features = vif_data[vif_data['VIF'] > 10]['feature'].tolist()

    # Drop collinear features and train the model accordingly
    X = X.drop(columns=[col for col in high_vif_features if col in X.columns], errors='ignore')

    # Split the cleaned data into training and testing sets again
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Normalize the cleaned data
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == '__main__':
    # Load data from raw data folder for a specific coin (e.g., BTCUSDT)
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'BTCUSDT.csv'))
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print("Preprocessing complete.")
