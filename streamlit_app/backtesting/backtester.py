import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, initial_balance=10000, position_size=0.1, 
                 stop_loss=0.02, take_profit=0.04):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def _calculate_position_size(self, balance):
        """Calculate position size based on current balance"""
        return balance * self.position_size
    
    def _check_stop_loss_take_profit(self, entry_price, current_price, position):
        """Check if stop loss or take profit has been hit"""
        if position == 0:
            return False, 0
            
        price_change = (current_price - entry_price) / entry_price
        if position > 0:  # Long position
            if price_change <= -self.stop_loss:
                return True, -self.stop_loss
            if price_change >= self.take_profit:
                return True, self.take_profit
        else:  # Short position
            if price_change >= self.stop_loss:
                return True, -self.stop_loss
            if price_change <= -self.take_profit:
                return True, self.take_profit
                
        return False, price_change
    
    def run(self, data, strategy):
        """Run backtest simulation"""
        balance = self.initial_balance
        position = 0  # 1 for long, -1 for short, 0 for no position
        entry_price = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            timestamp = data.index[i]
            
            # Track equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': balance,
                'price': current_price
            })
            
            # Check stop loss/take profit if in position
            if position != 0:
                closed, pnl = self._check_stop_loss_take_profit(
                    entry_price, current_price, position
                )
                if closed:
                    # Calculate actual PnL
                    position_value = self._calculate_position_size(balance)
                    trade_pnl = position_value * pnl
                    balance += trade_pnl
                    
                    trades.append({
                        'exit_time': timestamp,
                        'exit_price': current_price,
                        'pnl': trade_pnl,
                        'pnl_percent': pnl,
                        'exit_type': 'sl/tp',
                        'balance': balance
                    })
                    
                    position = 0
                    continue
            
            # Get new signal if not in position
            if position == 0:
                features = data.iloc[i][strategy.feature_columns]
                signal = strategy.predict(features)
                
                if signal != 0:  # New position
                    position = signal
                    entry_price = current_price
                    
                    trades.append({
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'position': 'long' if position > 0 else 'short',
                        'size': self._calculate_position_size(balance)
                    })
        
        # Close any open position at the end
        if position != 0:
            final_price = data['close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price
            if position < 0:
                pnl = -pnl
                
            position_value = self._calculate_position_size(balance)
            trade_pnl = position_value * pnl
            balance += trade_pnl
            
            trades.append({
                'exit_time': data.index[-1],
                'exit_price': final_price,
                'pnl': trade_pnl,
                'pnl_percent': pnl,
                'exit_type': 'close',
                'balance': balance
            })
        
        # Calculate metrics
        total_trades = len([t for t in trades if 'pnl' in t])
        winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
        
        metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': (balance - self.initial_balance) / self.initial_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return metrics
