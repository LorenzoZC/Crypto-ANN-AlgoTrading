import pandas as pd
from binance.client import Client
from tqdm import tqdm
import os

def get_binance_data(ticker, interval='4h', start='1 Jan 2018', end=None):
    client = Client()
    intervals = {
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h':  Client.KLINE_INTERVAL_1HOUR,      
        '4h':  Client.KLINE_INTERVAL_4HOUR,
        '1d':  Client.KLINE_INTERVAL_1DAY
    }
    interval = intervals.get(interval, Client.KLINE_INTERVAL_4HOUR)
    klines = client.get_historical_klines(symbol=ticker, interval=interval, start_str=start, end_str=end)
    data = pd.DataFrame(klines)
    data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    data.index = pd.to_datetime(data['open_time'], unit='ms')
    usecols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol']
    data = data[usecols]
    data.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'BaseAssetVolume', 'NumberOfTrades', 'TakerBuyVolume', 'TakerBuyBaseAssetVolume']
    data = data.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float', 'BaseAssetVolume': 'float', 'NumberOfTrades': 'int', 'TakerBuyVolume': 'float', 'TakerBuyBaseAssetVolume': 'float'})
    return data

def fetch_and_save_data(ticker_list, interval='1h', start='1 Jan 2018', end='30 Jun 2024'):
    client = Client()
    exchange_info = client.get_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
    # print('Number of active crypto pairs: ', len(symbols))
    print('Downloading data for these pairs: ', *ticker_list)

    # Create a dictionary to hold DataFrames for each coin
    coin_dataframes = {}

    for ticker in tqdm(ticker_list):
        try:
            coin_dataframes[ticker] = get_binance_data(ticker, interval=interval, start=start, end=end)
        except Exception as err:
            print(f"Error retrieving data for {ticker}: {err}")
            continue

    # Save each DataFrame to a separate CSV file in the data/raw directory
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    for ticker, df in coin_dataframes.items():
        df.to_csv(os.path.join(raw_data_dir, f'{ticker}.csv'), index=False)

    return coin_dataframes
