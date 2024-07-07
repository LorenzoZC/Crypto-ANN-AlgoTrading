class ML_Signal(bt.Strategy):
    
    params = (
        ('printlog', printlog),
        ('allocation', allocation),
        ('loglimit', 400),
        ('startdate', None),
        ('enddate', None),
        ('max_order_percentage', 0.05),
        ('max_drawdown', 0.10),
        ('cooldown_period', 10),
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
            slippage = self.broker.getcommissioninfo(self.data).get_slippage(
                order.executed.size, order.executed.price, self.data)
            if order.isbuy():
                self.log(
                    'BUY ORDER EXECUTED, Size: %.2f, Price: %.2f, Cost: %.2f, Commission: %.2f, Slippage: %.2f' %
                    (order.executed.size,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     abs(slippage*order.executed.size*order.executed.price/100)))
            if order.issell():
                self.log('SELL ORDER EXECUTED, Size: %.2f, Price: %.2f, Cost: %.2f, Commission: %.2f, Slippage: %.2f' %
                    (order.executed.size,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     abs(slippage*order.executed.size*order.executed.price/100)))

            self.total_commission += order.executed.comm
            self.bar_executed = len(self)
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.getstatusname()}')
            self.log_additional_details(order)
            self.order = None

        self.total_commission += order.executed.comm
        # Write down: no pending order
        self.order = None

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
        
    def manage_cash(self):
        cash = self.broker.get_cash()
        position_size = self.position.size
        position_price = self.dataclose[0] if self.position.price == 0 else self.position.price
        
        # Portfolio value calc
        if self.position.size >= 0:
            # Long or flat position
            portfolio_value = cash + (position_size * self.dataclose[0])
        else:
            # Short position
            portfolio_value = cash - (abs(position_size) * self.dataclose[0])

        if self.dataclose[0] == 0:
            self.log("Current close price is zero, skipping position adjustment to avoid division by zero.")
            return

        target_size = int((portfolio_value * self.params.allocation) / self.dataclose[0])
        target_value = target_size * self.dataclose[0]

        self.log(f"Calculated target_size: {target_size}, target_value: {target_value}")

        if self.datasignal[0] >= threshold:  # Long signal
            if position_size >= 0:  # Already long or flat
                if cash > self.dataclose[0]:  # Unused cash detected
                    buy_size = target_size - position_size # min(int(cash / self.dataclose[0]), target_size - position_size)
                    if buy_size > 0:
                        self.log(f"Investing unused cash: Buying {buy_size} units")
                        self.order = self.buy(size=buy_size)
            else:  # Short position
                self.log(f"Closing short position and going long")
                self.order = self.order_target_size(target=target_size, exectype=bt.Order.Market)
        else:  # Short signal
            if position_size <= 0:  # Already short or flat
                if cash > self.dataclose[0]:  # Unused cash detected
                    sell_size = abs(target_size) - abs(position_size) # min(int(cash / self.dataclose[0]), abs(target_size) - abs(position_size))
                    if sell_size > 0:
                        self.log(f"Investing unused cash: Selling {sell_size} units")
                        self.order = self.sell(size=sell_size)
            else:  # Long position
                self.log(f"Closing long position and going short")
                self.order = self.order_target_size(target=-abs(target_size), exectype=bt.Order.Market)

        self.log(f"Portfolio Value: {portfolio_value}, Available Cash: {cash}, "
                 f"Current Position: {position_size}, Target Size: {target_size}")
        
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('OPERATING PROFIT, GROSS: %.2f, NET: %.2f' % (trade.pnl, trade.pnlcomm))
            self.order = None  # Reset the order after trade is closed
        else:
            return
                
            ##################################Long/Short###############################################################
                
#     def next(self):
#         # Log closing price and prediction
#         self.log('Close, %.2f' % self.dataclose[0])
#         self.log('best_y_pred, %.2f' % self.datasignal[0])

#         # Check if an order is pending
#         if self.order:
#             self.log('Order pending... skipping')
#             return

#         # Log portfolio information
#         cash = self.broker.get_cash()
#         position_size = self.position.size
#         position_price = self.position.price
        
#         # Portfolio value calc
#         if self.position.size >= 0:
#             # Long or flat position
#             portfolio_value = cash + (position_size * self.dataclose[0])
#         else:
#             # Short position
#             portfolio_value = cash - (abs(position_size) * self.dataclose[0])
#         #self.log(f'Portfolio value: {portfolio_value:.2f}')
        
#         self.log(f'Available cash: {cash:.2f}, Current position size: {position_size}, Cost price: {position_price:.2f}')

#         # Determine the target position based on the signal
#         target_size = int((portfolio_value * self.params.allocation) / self.dataclose[0]) 
#         if self.datasignal[0] >= threshold:
#             target_size = abs(target_size)  # Long position
#         else:
#             target_size = -abs(target_size)  # Short position

#         size_diff = target_size - self.position.size
        
#         # if target_size * self.dataclose[0] > portfolio_value:
#         #     print("Target position value exceeds portfolio value")
#         # if abs(position_size) > target_size and cash >= self.dataclose[0]:
#         #     print("Unused cash detected")
#         self.log(f"Target size: {target_size}, Current position: {position_size}, Available cash: {cash}")
        
#         available_liquidity = self.hourly_volume[0]
#         max_order_size = available_liquidity * self.params.max_order_percentage
#         adjusted_target_size = min(abs(size_diff), max_order_size)
#         # adjusted_target_size = min(abs(size_diff), max_order_size) * (1 if size_diff > 0 else -1)
        
#         if size_diff != 0:
#             # Close existing position and open new position in one step
#             self.log(f'Signal: {"LONG" if self.datasignal[0] >= threshold else "SHORT"}')
#             self.log(f'Adjusting position from {self.position.size} to {target_size}')

#             if self.position.size == 0:
#                 if target_size > 0:
#                     self.log(f'BUY ORDER CREATED, Size: {target_size}, Price: {self.dataclose[0]:.2f}')
#                 else:
#                     self.log(f'SELL ORDER CREATED, Size: {abs(target_size)}, Price: {self.dataclose[0]:.2f}')
#             elif self.position.size > 0 and target_size < 0:
#                 self.log(f'SELL ORDER CREATED TO CLOSE LONG AND GO SHORT, Size: {abs(target_size)}, Price: {self.dataclose[0]:.2f}')
#             elif self.position.size < 0 and target_size > 0:
#                 self.log(f'BUY ORDER CREATED TO CLOSE SHORT AND GO LONG, Size: {target_size}, Price: {self.dataclose[0]:.2f}')
#             else:
#                 if target_size > self.position.size:
#                     self.log(f'BUY ORDER CREATED TO INCREASE POSITION, Size: {target_size - self.position.size}, Price: {self.dataclose[0]:.2f}')
#                 else:
#                     self.log(f'SELL ORDER CREATED TO DECREASE POSITION, Size: {self.position.size - target_size}, Price: {self.dataclose[0]:.2f}')

#             if not self.order:
#                 self.order = self.order_target_size(target=adjusted_target_size, exectype=bt.Order.Market)
#         else:
#             self.log('No change in position needed')
            
#         self.manage_cash()

##################################Long only###############################################################
        
    def next(self):
        
        # Log closing price and prediction
        self.log('Close, %.2f' % self.dataclose[0])
        self.log('best_y_pred, %.2f' % self.datasignal[0])

        # Check if an order is pending
        if self.order:
            self.log('Order pending... skipping')
            return

        # Log portfolio information
        cash = self.broker.get_cash()
        position_size = self.position.size
        position_price = self.position.price

        # Portfolio value calc (Long only)
        portfolio_value = cerebro.broker.getvalue() # cash + (position_size * self.dataclose[0])

        self.log(f'Available cash: {cash:.2f}, Current position size: {position_size}, Cost price: {position_price:.2f}')
        self.log(f'Portfolio value: {portfolio_value:.2f}')
        
        # Implement cooldown period after drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
            
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        if drawdown >= self.params.max_drawdown:
            self.cooldown_end_date = self.datas[0].datetime.datetime(0) + timedelta(days=self.params.cooldown_period)
            self.max_portfolio_value = portfolio_value  # Reset max portfolio value at the end of the cooldown
            self.log(f'Drawdown of {drawdown*100:.2f}% detected. Cooling down until {self.cooldown_end_date}')
            # self.log(f'CLOSING LONG POSITION, Size: {self.position.size}, Price: {self.dataclose[0]:.2f}')
            # self.order = self.close()
        
        # Check if in cooldown period
        if self.cooldown_end_date and self.datas[0].datetime.datetime(0) < self.cooldown_end_date:
            self.log('In cooldown period. No trading.')
            return
        
        # Define position size
        
        target_size = round((portfolio_value * self.params.allocation) / self.dataclose[0], 2)# int((portfolio_value * self.params.allocation) / self.dataclose[0])
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
        adjusted_target_size = min(abs(size_diff), max_order_size)

        # Open long position when signal exceeds threshold
        if self.datasignal[0] >= self.threshold and position_size == 0:
            self.log(f'BUY ORDER CREATED, Size: {adjusted_target_size}, Price: {self.dataclose[0]:.2f}')
            self.order = self.buy(target=adjusted_target_size)#, exectype=bt.Order.Market)

        # Close long position when signal falls below threshold
        elif self.datasignal[0] < self.threshold and position_size > 0:
            self.log(f'SELL ORDER CREATED, Size: {abs(position_size)}, Price: {self.dataclose[0]:.2f}')
            self.order = self.close()

        # No order if already in long position or signal below threshold
        else:
            self.log('No change in position needed')

          # Manage cash (optional)
          # self.manage_cash()
