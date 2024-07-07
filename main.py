import backtrader as bt
import pandas as pd
import pytz
from datetime import datetime
from strategies.ml_signal import ML_Signal
from analyzers.commission_analyzer import CommissionAnalyzer

# Initialize cerebro
cerebro = bt.Cerebro(cheat_on_open=True)

# Load data
clean_df = pd.read_csv('data/clean_data.csv', index_col='OpenTime', parse_dates=True)

# Add data to cerebro
data_bt = bt.feeds.PandasDirectData(dataname=clean_df)
cerebro.adddata(data_bt)

# Set up the strategy
threshold = 0.5
printlog = True
allocation = 0.9
startdate = datetime(2021, 4, 1, tzinfo=pytz.UTC)
enddate = datetime(2021, 4, 7, tzinfo=pytz.UTC)

cerebro.addstrategy(ML_Signal, printlog=printlog, startdate=startdate, enddate=enddate, threshold=threshold, allocation=allocation)

# Set the initial cash
starting_cash = 100000.0
cerebro.broker.setcash(starting_cash)

# Add the PortfolioSizer with a 90% allocation
cerebro.addsizer(bt.sizers.FixedSize, stake=1)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', timeframe=bt.TimeFrame.Days, compression=1)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='draw_down')
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analysis')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Value at Risk
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number
cerebro.addanalyzer(CommissionAnalyzer, _name='commission')

# Run the backtest
results = cerebro.run()
strat = results[0]

# Print the final result
print('\nFinal Portfolio Value: %.2f' % cerebro.broker.getvalue())

##################

import backtrader as bt
from strategies.ml_signal_strategy import ML_Signal
from utils.data_fetcher import fetch_data
from utils.data_preprocessor import preprocess_data
from utils.logger import setup_logger

def run_strategy():
    # Fetch and preprocess data
    raw_data = fetch_data()
    data = preprocess_data(raw_data)

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

if __name__ == '__main__':
    run_strategy()
