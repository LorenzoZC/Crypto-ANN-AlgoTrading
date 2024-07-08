import backtrader as bt
import pandas as pd
import pytz
from datetime import datetime
import os

from strategies.base_strategy import *
from models.training import *
from analyzers.commission_analyzer import CommissionAnalyzer
from utils.data_fetcher import fetch_and_save_data
from utils.data_preprocessor import preprocess_data
from utils.logger import setup_logger
from utils.performance_metrics import *

is_cv = False

allocation = 0.99
threshold = 0.5

# Logging
printlog = False
startdate = datetime(2021, 4, 6)
enddate = datetime(2024, 1, 1)

coin = 'BTC'

##################

def run_strategy():
    # Define ticker list and fetch data
    ticker_list = ['BTCUSDT', 'ETHUSDT']
    coin_dataframes = fetch_and_save_data(ticker_list)
    
    # Process data for a specific coin (e.g., BTC)
    coin = coin
    data = coin_dataframes.get(f'{coin}USDT')
    
    if data is not None:
        # Further processing if needed
        data['OpenTime'] = pd.to_datetime(data['OpenTime'])
        data['CloseTime'] = pd.to_datetime(data['CloseTime'])
        data['Close'] = data['Close'].astype(float)
        data = data.dropna(subset=['Close'])

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        lstm_X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        lstm_X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model_trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        clean_df = model_trainer.train_models(X_train, X_test, y_train, y_test)

        output_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'clean_df.csv')
        clean_df.to_csv(output_path)
        print(f"Cleaned data saved to {output_path}")

        #############################################################################################
        
        cerebro = bt.Cerebro(cheat_on_open=True) #
        cerebro.addstrategy(ML_Signal, printlog=printlog, startdate=startdate, enddate=enddate)

        data_bt = bt.feeds.PandasDirectData(dataname=clean_df, fromdate=startdate, todate=enddate)
        cerebro.adddata(data_bt)

        starting_cash = 100000.0
        cerebro.broker.setcash(starting_cash)

        # Add leverage to the broker
        commission_info = CommissionSlippageLeverage()
        cerebro.broker.setcommission(commission=0, leverage=commission_info.params.leverage)
        cerebro.broker.addcommissioninfo(commission_info)
        cerebro.addsizer(PortfolioSizer)#, allocation=allocation)

        # Print out the starting conditions
        print(f"The following transactions show the backtesting results of {coin}:")
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        print()

        # Analyzer
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', timeframe=bt.TimeFrame.Days, compression=1)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='draw_down')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analysis')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Value at Risk
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number
        cerebro.addanalyzer(CommissionAnalyzer, _name='commission')

        results = cerebro.run()
        strat = results[0]

        # Print out the final result
        print('\nFinal Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Performance metrics calculation
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

        bmk_ticker = coin
        metrics = calculate_performance_metrics(returns, bmk_ticker, startdate, enddate)
                
        print(f'The return series metrics are calculated between {startdate} and {enddate}')
        print()
        print(f'Performance Metrics for {coin}:')
        print('----------------------------------------')
        print('Cumulative Return:', round(metrics['cumulative_return'], 2))
        print('Annual Return:', round(metrics['annual_return'], 2))
        print('Sharpe Ratio:', round(metrics['sharpe_ratio'], 2))
        print('Sortino Ratio:', round(metrics['sortino_ratio'], 2))
        print('Skew:', round(metrics['skewness'], 2))
        print('Kurtosis:', round(metrics['kurtosis'], 2))
        print('Tail Ratio:', round(metrics['tail_ratio'], 2))
        print('Daily Value at Risk (VaR):', f'{metrics['daily_var']:.2%}' if not np.isnan(metrics['daily_var']) else "NaN")
        print('Maximum DrawDown:', round(metrics['max_drawdown'], 2), '%')
        print('--------------------------------')
        print()

        print('Benchmark Performance Metrics:')
        print('--------------------------------')
        print('Benchmark Cumulative Return:', round(metrics['benchmark_cumulative_return'], 2))
        print('Benchmark Annual Return:', round(metrics['benchmark_annual_return'], 2))
        print('Benchmark Sharpe Ratio:', round(metrics['benchmark_sharpe_ratio'], 2))
        print('Benchmark Sortino Ratio:', round(metrics['benchmark_sortino_ratio'], 2))
        print('Benchmark Skew:', round(metrics['benchmark_skewness'], 2))
        print('Benchmark Kurtosis:', round(metrics['benchmark_kurtosis'], 2))
        print('Benchmark Tail Ratio:', round(metrics['benchmark_tail_ratio'], 2))
        print('Benchmark Daily Value at Risk (VaR):', f'{metrics['benchmark_daily_var']:.2%}' if not np.isnan(metrics['benchmark_daily_var']) else "NaN")
        print('Benchmark Maximum DrawDown:', round(metrics['benchmark_max_drawdown'], 2), '%')
        print('--------------------------------')
        print()
        
    else:
        print(f"Data for {coin} not found.")

if __name__ == '__main__':
    run_strategy()
