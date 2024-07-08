import backtrader as bt
from datetime import datetime, timedelta
is_cv = False

allocation = 0.99 # to avoid margin issues
threshold = 0.5

# Logging
printlog = False
startdate = datetime(2021, 4, 6)
enddate = datetime(2024, 1, 1)

class CommissionSlippageLeverage(bt.CommInfoBase):
    params = (
        ('base_slippage', 0.05/100),
        ('max_slippage', 0.5/100),
        ('volatility_threshold', 0.30),
        ('commission', 0.1),
        ('leverage', 1.00),
        ('leverage_cost_daily', 0.0002),  # Daily leverage cost rate
    )

    def __init__(self):
        super().__init__()
        self.strat = strat # reference the strategy

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
        # self.strat.log(f"Calculated slippage: {slippage}")
        return slippage

    def getoperationcost(self, size, price, data):
        slippage = self.get_slippage(size, price, data)
        slippage_cost = size * price * slippage
        commission_cost = size * price * self.params.commission
        leverage_cost = size * price * self.params.leverage_cost_daily if self.params.leverage >= 1.000001 else 0
        total_cost = slippage_cost + commission_cost + leverage_cost
        self.strat.log(f"Slippage cost: {slippage_cost}, Commission cost: {commission_cost}, Leverage cost: {leverage_cost}, Total cost: {total_cost}")  # Added print statements for debugging
        return total_cost

class ML_Signal(bt.Strategy):
    
    params = (
        ('printlog', printlog),
        ('allocation', allocation),
        ('loglimit', 400),
        ('startdate', None),
        ('enddate', None),
        ('max_order_percentage', 0.01),
        ('max_drawdown', 0.10),
        ('cooldown_period', 5),
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        dt = dt or self.datas[0].datetime.datetime(0)
        # if self.log_count >= self.params.loglimit:
        #     return
        if self.params.startdate and dt < self.params.startdate:
            return
        if self.params.enddate and dt > self.params.enddate:
            return
        if self.params.printlog:
            print('%s, %s' % (dt.isoformat(), txt))
        self.log_count += 1

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datasignal = self.datas[0].openinterest
        self.hourly_volume = self.datas[0].volume
        self.threshold = threshold
        
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        self.total_commission = 0
        self.last_id = 0
        self.log_count = 0
        
        self.max_portfolio_value = self.broker.getvalue()
        self.cooldown_end_date = None
        
        self.cooldown_counter = 0
        self.cooldown_period = 3
        
        self.open_commission = 0
        self.close_commission = 0
        self.in_position = False
        
        self.trade_start_cash = None
        
    def get_id(self):
        self.last_id += 1
        return self.last_id

    def notify_order(self, order):
        if order.status in [order.Submitted]:
            self.log('ORDER SUBMITTED')
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        if order.status in [order.Accepted]:
            self.log('ORDER ACCEPTED')
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            
            cash = self.broker.getcash()
            
            # self.log(f"Cash before update: {cash}")
            
            # commission_info = CommissionSlippageLeverage() # self.broker.getcommissioninfo(self.data)
            # total_cost = round(commission_info.getoperationcost(order.executed.size, order.executed.price, self.data),2)
            
            commission = order.executed.comm
        
            # self.log(f"Total cost: {commission}")
            # self.broker.setcash(cash-commission) # (self.broker.getcash() - total_cost)
            # self.log(f"Cash after update: {self.broker.getcash()}")
            
            slippage = commission_info.get_slippage(order.executed.size, order.executed.price, self.data)
            
            if not self.in_position:
                self.open_commission = commission
                self.in_position = True
                self.trade_start_cash = self.broker.getcash() 
            else:
                self.close_commission = commission
                self.in_position = False
            
            if order.isbuy():
                self.log(
                    'BUY ORDER EXECUTED, Size: %.2f, Price: %.2f, Cash impact: %.2f, Cost: %.2f, Commission: %.2f, Slippage: %.2f, Cash: %.2f' %
                    (order.executed.size,
                     order.executed.price,
                     -order.executed.size*order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     slippage, # abs(slippage*order.executed.size*order.executed.price)
                     cash
                    ))
            if order.issell():
                self.log('SELL ORDER EXECUTED, Size: %.2f, Price: %.2f, Revenue: %.2f, Cost: %.2f, Commission: %.2f, Slippage: %.2f, Cash: %.2f' %
                    (order.executed.size,
                     order.executed.price,
                     -order.executed.size*order.executed.price,
                     -order.executed.value,
                     order.executed.comm,
                     slippage, # abs(slippage*order.executed.size*order.executed.price)
                     cash
                    ))

            self.total_commission += order.executed.comm
            self.bar_executed = len(self)
            self.order = None # Write down: no pending order

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.getstatusname()}')
            self.log_additional_details(order)
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            
            cash = self.broker.getcash()
            gross_profit = trade.pnl
            net = trade.pnlcomm
            cost = trade.pnl-trade.pnlcomm
            commission = trade.commission
            slippage = cost-commission  # Extract slippage from total cost
            net_profit = gross_profit - commission - slippage
            opencomm = self.open_commission
            closecomm = self.close_commission
            total_order_commission = opencomm + closecomm
            
            # self.log(f'Open Order Commission: {opencomm:.2f}')
            # self.log(f'Close Order Commission: {closecomm:.2f}')
            # self.log(f'Total Order Commissions: {total_order_commission:.2f}')
            # self.log(f'Trade Commission (from trade object): {commission:.2f}')
            
            self.log('OPERATING PROFIT, GROSS: %.2f, NET: %.2f' % (gross_profit, net))
            self.log('Commission paid, open: %.2f, close: %.2f, total: %.2f' % (opencomm, closecomm, total_order_commission))
            
            if abs((total_order_commission - commission)) > 0.01:
                self.log(f'WARNING: Discrepancy in commission calculations')

            # Reset
            self.open_commission = 0
            self.close_commission = 0
            self.trade_start_cash = None
            
            self.order = None  # Reset the order after trade is closed
        else:
            return
    
    def log_additional_details(self, order):
        cash = self.broker.get_cash()
        position_size = self.position.size
        position_price = self.position.price
        portfolio_value = cash + position_size * self.dataclose[0]

        self.log(f'Order ID: {order.ref}')
        self.log(f'Order Type: {"Buy" if order.isbuy() else "Sell"}')
        self.log(f'Order Size: {order.size}')
        self.log(f'Order Price: {order.created.price if order.created.price else "Market Order"}')
        self.log(f'Order Cost: {order.created.value}')
        self.log(f'Available Cash: {cash:.2f}')
        self.log(f'Current Position Size: {position_size}')
        self.log(f'Current Position Price: {position_price:.2f}')
        self.log(f'Portfolio Value: {portfolio_value:.2f}')
        self.log(f'Order Status: {order.getstatusname()}')

    # Long when signal > 0.5, 0 otherwise
        
    def next(self):

        # Check if an order is pending
        if self.order:
            self.log('Order pending... skipping')
            return

        # Log portfolio information
        cash = self.broker.get_cash()
        position_size = self.position.size
        position_price = self.position.price
        
        previous_portfolio_value = getattr(self, 'previous_portfolio_value', None)
        portfolio_value_discrepancy = getattr(self, 'portfolio_value_discrepancy', False)  # Retrieve discrepancy flag
        
        # Set portfolio value and correct backtrader error that drags it down randomly to be = cash randomly
        if cash == self.broker.getvalue():  # Check for discrepancy again
            if not portfolio_value_discrepancy:  # No previous discrepancy
                portfolio_value = self.broker.getvalue()  # Use current value
            else:
                portfolio_value = previous_portfolio_value  # Use previous value if discrepancy occurred before
        else:
            portfolio_value = self.broker.getvalue()  # Assume correct value if cash and getvalue differ
            portfolio_value_discrepancy = False  # Reset discrepancy flag since cash and getvalue differ
            
        # Update previous portfolio value and discrepancy flag for next iteration
        self.previous_portfolio_value = portfolio_value
        setattr(self, 'portfolio_value_discrepancy', portfolio_value_discrepancy)

        # # Portfolio value calc (Long only)
        # portfolio_value = cerebro.broker.getvalue() # cash + (position_size * self.dataclose[0]) # 

        self.log('Open, %.2f' % self.dataopen[0])
        self.log(f'Available cash: {cash:.2f}, Current position size: {position_size}, Cost price: {position_price:.2f}')
        self.log(f'Portfolio value: {portfolio_value:.2f}')
        self.log('Close, %.2f' % self.dataclose[0])
        self.log('best_y_pred, %.2f' % self.datasignal[0])
        
        # Implement cooldown period after drawdown
        if portfolio_value <= cash: 
            pass # backtrader shows random instances where portfolio_value drops to cash for one period then returns to normal, but this triggers the cooldown when it shouldn't
        else:
            if portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = portfolio_value

            drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            if drawdown >= self.params.max_drawdown:
                self.cooldown_end_date = self.datas[0].datetime.datetime(0) + timedelta(days=self.params.cooldown_period)
                self.max_portfolio_value = portfolio_value  # Reset max portfolio value at the end of the cooldown
                self.log(f'Drawdown of {drawdown*100:.2f}% detected. Cooling down until {self.cooldown_end_date}')

            # Check if in cooldown period
            if self.cooldown_end_date and self.datas[0].datetime.datetime(0) < self.cooldown_end_date:
                self.log('In cooldown period. No trading.')
                return
        
        # Define position size
        
        target_size = round((portfolio_value * self.params.allocation) / self.dataclose[0], 2)
        size_diff = target_size - self.position.size
        
        # Limit volume to a % of avg volume to reflect liquidity
        total_hourly_volume = 0
        available_data = min(len(self.hourly_volume), 24)  # Limit to available data

        if available_data > 0:
            for i in range(available_data):
                total_hourly_volume += self.hourly_volume[-(i+1)]  # Access from most recent to oldest
            average_hourly_volume = total_hourly_volume / available_data
        else:
            average_hourly_volume = self.hourly_volume[0]
            
        available_liquidity = average_hourly_volume
        max_order_size = available_liquidity * self.params.max_order_percentage
        adjusted_target_size = round(min(abs(size_diff), max_order_size),2)
        
        # self.log(f'Liquidity capped at: {max_order_size}')
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Open long position when signal exceeds threshold
        if self.datasignal[0] >= self.threshold and position_size == 0 and self.cooldown_counter == 0:
            self.log(f'BUY ORDER CREATED, Size: {adjusted_target_size}, Price: {self.dataopen[1]:.2f}')
            self.order = self.buy(size=adjusted_target_size)#, exectype=bt.Order.Market)

        # Close long position when signal falls below threshold
        elif self.datasignal[0] < self.threshold and position_size > 0:
            self.log(f'SELL ORDER CREATED, Size: {abs(position_size)}, Price: {self.dataclose[0]:.2f}')
            self.order = self.close()
            # Start cooldown period
            self.cooldown_counter = self.cooldown_period

        # No order if already in long position or signal below threshold
        else:
            self.log('No change in position needed')
            
        # Log cooldown status
        if self.cooldown_counter > 0:
            self.log(f'In cooldown period. {self.cooldown_counter} periods remaining.')
    
class PortfolioSizer(bt.Sizer):
    params = (
        ('allocation', allocation),  
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        size = int((cash * self.params.allocation) / data.close[0])
        return size

    # For Long Short
    
    # def _getsizing(self, comminfo, cash, data, isbuy):
    #     if isbuy:
    #         size = int((cash * self.params.allocation) / data.close[0])
    #     else:
    #         size = self.broker.getposition(data).size
    #     return size
    
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
