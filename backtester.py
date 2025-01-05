import pandas as pd
import numpy as np
from data_fetcher import OANDADataFetcher
from strategy import MLStrategy

class Backtester:
    def __init__(self, initial_balance=10000, position_size=0.1):
        """
        initial_balance: Starting account balance
        position_size: Fraction of balance to risk per trade (0.1 = 10%)
        """
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.reset()
        
    def reset(self):
        """Reset the backtester state"""
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []
        self.current_position = None
        
    def calculate_position_size(self):
        """Calculate position size based on current balance"""
        return self.balance * self.position_size
        
    def execute_trade(self, price, signal, timestamp):
        """Execute a trade based on the signal"""
        if signal == 0:  # No signal
            return
            
        position_size = self.calculate_position_size()
        
        if self.current_position is None and signal in [-1, 1]:
            # Open new position
            self.current_position = {
                'type': 'long' if signal == 1 else 'short',
                'entry_price': price,
                'size': position_size,
                'entry_time': timestamp
            }
            self.positions.append(self.current_position)
            
        elif self.current_position is not None:
            # Close existing position
            exit_price = price
            pnl = 0
            
            if self.current_position['type'] == 'long':
                pnl = (exit_price - self.current_position['entry_price']) * self.current_position['size']
            else:  # short
                pnl = (self.current_position['entry_price'] - exit_price) * self.current_position['size']
                
            self.balance += pnl
            
            self.trades.append({
                'entry_time': self.current_position['entry_time'],
                'exit_time': timestamp,
                'type': self.current_position['type'],
                'entry_price': self.current_position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'balance': self.balance
            })
            
            self.current_position = None
            
    def run(self, data, strategy):
        """Run backtest on historical data"""
        self.reset()
        
        for timestamp, row in data.iterrows():
            features = row[strategy.feature_columns]
            signal = strategy.predict(features)
            self.execute_trade(row['close'], signal, timestamp)
            
        # Close any open position at the end
        if self.current_position is not None:
            self.execute_trade(data['close'].iloc[-1], 0, data.index[-1])
            
        return self.generate_statistics()
        
    def generate_statistics(self):
        """Calculate backtest statistics"""
        if not self.trades:
            return "No trades executed"
            
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = trades_df['pnl'].sum()
        max_drawdown = self.calculate_max_drawdown(trades_df)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'max_drawdown_pct': max_drawdown
        }
        
    def calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown percentage"""
        if trades_df.empty:
            return 0
            
        balances = trades_df['balance'].values
        peak = self.initial_balance
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown

if __name__ == '__main__':
    # Test the backtester
    fetcher = OANDADataFetcher()
    data = fetcher.fetch_historical_data(days=90)
    data_with_features = fetcher.add_features(data)
    
    strategy = MLStrategy()
    strategy.train(data_with_features)
    
    backtester = Backtester(initial_balance=10000, position_size=0.1)
    results = backtester.run(data_with_features, strategy)
    
    print("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
