import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, initial_balance=10000, position_size=0.1):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
    
    def run(self, data, strategy):
        """Run backtest on historical data"""
        self.reset()
        
        for i in range(len(data)-1):
            current_data = data.iloc[i]
            next_data = data.iloc[i+1]
            
            # Get trading signal
            signal = strategy.predict(current_data)
            confidence = strategy.get_prediction_confidence(current_data)
            
            # Execute trades based on signal
            if signal != 0 and confidence >= 0.6:  # Only trade with sufficient confidence
                if self.position is None:  # No position, consider entering
                    if signal == 1:  # Buy signal
                        self._enter_long(current_data, next_data)
                    elif signal == -1:  # Sell signal
                        self._enter_short(current_data, next_data)
                else:  # In position, consider exiting
                    if (self.position['type'] == 'long' and signal == -1) or \
                       (self.position['type'] == 'short' and signal == 1):
                        self._exit_position(current_data, next_data)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': current_data.name,
                'equity': self._calculate_equity(current_data['close'])
            })
        
        # Close any remaining position
        if self.position is not None:
            self._exit_position(data.iloc[-2], data.iloc[-1])
        
        return self._generate_results()
    
    def _enter_long(self, current_data, next_data):
        """Enter long position"""
        entry_price = next_data['open']
        position_size = (self.balance * self.position_size) / entry_price
        
        self.position = {
            'type': 'long',
            'entry_price': entry_price,
            'size': position_size,
            'entry_time': next_data.name
        }
    
    def _enter_short(self, current_data, next_data):
        """Enter short position"""
        entry_price = next_data['open']
        position_size = (self.balance * self.position_size) / entry_price
        
        self.position = {
            'type': 'short',
            'entry_price': entry_price,
            'size': position_size,
            'entry_time': next_data.name
        }
    
    def _exit_position(self, current_data, next_data):
        """Exit current position"""
        exit_price = next_data['open']
        
        # Calculate profit/loss
        if self.position['type'] == 'long':
            pnl = (exit_price - self.position['entry_price']) * self.position['size']
        else:  # short
            pnl = (self.position['entry_price'] - exit_price) * self.position['size']
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        self.trades.append({
            'type': self.position['type'],
            'entry_time': self.position['entry_time'],
            'exit_time': next_data.name,
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'return': pnl / (self.position['entry_price'] * self.position['size'])
        })
        
        self.position = None
    
    def _calculate_equity(self, current_price):
        """Calculate current equity including open position value"""
        if self.position is None:
            return self.balance
        
        if self.position['type'] == 'long':
            unrealized_pnl = (current_price - self.position['entry_price']) * self.position['size']
        else:  # short
            unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
        
        return self.balance + unrealized_pnl
    
    def _generate_results(self):
        """Generate backtest results and statistics"""
        if not self.trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': self.initial_balance
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        avg_return = trades_df['return'].mean()
        
        # Calculate max drawdown
        equity_df['drawdown'] = equity_df['equity'].cummax() - equity_df['equity']
        max_drawdown = equity_df['drawdown'].max() / equity_df['equity'].cummax().max()
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        return {
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
