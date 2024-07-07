import backtrader as bt
import pandas as pd
import pytz
from datetime import datetime
from strategies.ml_signal import ML_Signal
from analyzers.commission_analyzer import CommissionAnalyzer
from utils.data_fetcher import fetch_and_save_data
from utils.data_preprocessor import preprocess_data
from utils.logger import setup_logger
import os

##################

def run_strategy():
    # Define ticker list and fetch data
    ticker_list = ['BTCUSDT', 'ETHUSDT']
    coin_dataframes = fetch_and_save_data(ticker_list)
    
    # Process data for a specific coin (e.g., BTC)
    coin = 'BTC'
    data = coin_dataframes.get(f'{coin}USDT')
    
    if data is not None:
        # Further processing if needed
        data['OpenTime'] = pd.to_datetime(data['OpenTime'])
        data['CloseTime'] = pd.to_datetime(data['CloseTime'])
        data['Close'] = data['Close'].astype(float)
        data = data.dropna(subset=['Close'])

        # Set up Cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(ML_Signal)

        # Add data to Cerebro
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)

        # Set initial capital
        cerebro.broker.set_cash(100000.0)

        # Set commission
        cerebro.broker.setcommission(commission=0.001)

        # Run strategy
        results = cerebro.run()

        # Log final results
        final_value = cerebro.broker.getvalue()
        logger = setup_logger()
        logger.info(f'Final Portfolio Value: {final_value:.2f}')
    else:
        print(f"Data for {coin} not found.")

if __name__ == '__main__':
    run_strategy()
