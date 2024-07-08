is_cv = False

allocation = 0.99
threshold = 0.5

# Logging
printlog = False

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

startdate = datetime(2021, 4, 6)
enddate = datetime(2024, 1, 1)

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

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        lstm_X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        lstm_X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model_trainer = ModelTrainer(is_cv=False)
        clean_df = model_trainer.train_models(X_train, X_test, y_train, y_test)

        output_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'clean_df.csv')
        clean_df.to_csv(output_path)
        print(f"Cleaned data saved to {output_path}")

        #############################################################################################
        
        class CommissionSlippageLeverage(bt.CommInfoBase):
            params = (
                ('base_slippage', 0.05),
                ('max_slippage', 0.5),
                ('volatility_threshold', 0.30),
                ('commission', 0.1),
                ('leverage', 1.0),
                ('leverage_cost_daily', 0.0002),  # Daily leverage cost rate
            )
        
            def __init__(self):
                super().__init__()
        
            def get_slippage(self, size, price, data):
                returns = np.diff(np.log(data.close.get(size=20)))
                if len(returns) <= 1:
                    return self.params.base_slippage
                volatility = np.std(returns)
                if volatility <= self.params.volatility_threshold:
                    slippage = self.params.base_slippage
                else:
                    # Ratio of how much volatility exceeds the threshold
                    excess_ratio = (volatility - self.params.volatility_threshold) / (self.params.max_slippage - self.params.volatility_threshold)
                    # Ensure the ratio does not exceed 1
                    excess_ratio = min(excess_ratio, 1.0)
                    # Interpolate between base slippage and max slippage based on the excess ratio
                    slippage = self.params.base_slippage + excess_ratio * (self.params.max_slippage - self.params.base_slippage)
                return slippage
        
            def getoperationcost(self, size, price, data):
                slippage = self.get_slippage(data)
                slippage_cost = size * price * slippage
                commission_cost = size * price * self.params.commission
        
                # Daily leverage cost calculation
                leverage_cost = size * price * self.params.leverage_cost_daily if self.params.leverage > 1 else 0
        
                total_cost = slippage_cost + commission_cost + leverage_cost
                return total_cost
            
        class PortfolioSizer(bt.Sizer):
            params = (
                ('allocation', allocation),  
            )
        
            # def _getsizing(self, comminfo, cash, data, isbuy):
            #     if isbuy:
            #         size = int((cash * self.params.allocation) / data.close[0])
            #     else:
            #         size = self.broker.getposition(data).size
            #     return size
            
            def _getsizing(self, comminfo, cash, data, isbuy):
                size = int((cash * self.params.allocation) / data.close[0])
                return size
            
        class CommissionAnalyzer(bt.Analyzer):
            def __init__(self):
                self.total_commission = 0
        
            def notify_trade(self, trade):
                if trade.isclosed:
                    self.total_commission += trade.commission
        
            def get_analysis(self):
                return {
                    'total_commission': self.total_commission
                }
        
        cerebro = bt.Cerebro(cheat_on_open=True) #
        cerebro.addstrategy(ML_Signal, printlog=printlog, startdate=startdate, enddate=enddate)
        
        data_bt = bt.feeds.PandasDirectData(dataname=clean_df)
        cerebro.adddata(data_bt)
        
        starting_cash = 100000.0
        cerebro.broker.setcash(starting_cash)
        
        # Add leverage to the broker
        cerebro.broker.setcommission(commission=0.0, leverage=CommissionSlippageLeverage.params.leverage)
        
        cerebro.addsizer(PortfolioSizer)#, allocation=allocation)
        
        # Set the commission, slippage, and leverage
        commission_info = CommissionSlippageLeverage()
        cerebro.broker.addcommissioninfo(commission_info)
        
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
    else:
        print(f"Data for {coin} not found.")

if __name__ == '__main__':
    run_strategy()
