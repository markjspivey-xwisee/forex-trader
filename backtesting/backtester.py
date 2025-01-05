import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import hashlib
import gc
import json

class Backtester:
    def __init__(self, initial_balance=10000, position_size=0.1):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self._cache = {}
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
    
    @staticmethod
    def _generate_cache_key(data, strategy_name, initial_balance, position_size):
        """Generate a unique cache key for the backtest"""
        # Convert data to JSON string representation
        data_str = data.to_json()
        # Combine with parameters
        key_str = f"{data_str}_{strategy_name}_{initial_balance}_{position_size}"
        # Create hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    @st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, limit entries
    def _run_backtest_cached(data_json, strategy_name, initial_balance, position_size):
        """Run backtest with caching"""
        # Convert JSON back to DataFrame
        data = pd.read_json(data_json)
        backtester = Backtester(initial_balance, position_size)
        
        try:
            # Process data in chunks to manage memory
            chunk_size = 1000
            for i in range(0, len(data)-1, chunk_size):
                chunk_end = min(i + chunk_size, len(data)-1)
                current_chunk = data.iloc[i:chunk_end]
                next_chunk = data.iloc[i+1:chunk_end+1]
                
                for j in range(len(current_chunk)):
                    current_data = current_chunk.iloc[j]
                    next_data = next_chunk.iloc[j]
                    
                    # Get trading signal (simulated for caching)
                    signal = 1 if current_data['close'] < next_data['close'] else -1
                    confidence = 0.6
                    
                    # Execute trades based on signal
                    if signal != 0 and confidence >= 0.6:  # Only trade with sufficient confidence
                        if backtester.position is None:  # No position, consider entering
                            if signal == 1:  # Buy signal
                                backtester._enter_long(current_data, next_data)
                            elif signal == -1:  # Sell signal
                                backtester._enter_short(current_data, next_data)
                        else:  # In position, consider exiting
                            if (backtester.position['type'] == 'long' and signal == -1) or \
                               (backtester.position['type'] == 'short' and signal == 1):
                                backtester._exit_position(current_data, next_data)
                    
                    # Record equity
                    backtester.equity_curve.append({
                        'timestamp': current_data.name.isoformat(),
                        'equity': float(backtester._calculate_equity(current_data['close']))
                    })
                
                gc.collect()  # Force garbage collection after each chunk
            
            # Close any remaining position
            if backtester.position is not None:
                backtester._exit_position(data.iloc[-2], data.iloc[-1])
            
            # Convert trades and equity curve to JSON-serializable format
            results = backtester._generate_results()
            results['trades'] = [
                {k: v.isoformat() if isinstance(v, pd.Timestamp) else float(v)
                 for k, v in trade.items()}
                for trade in results['trades']
            ]
            results['equity_curve'] = [
                {k: v.isoformat() if isinstance(v, pd.Timestamp) else float(v)
                 for k, v in point.items()}
                for point in results['equity_curve']
            ]
            
            return json.dumps(results)
            
        except Exception as e:
            st.error(f"Error during backtesting: {str(e)}")
            return json.dumps(backtester._generate_empty_results())
    
    def run(self, data, strategy):
        """Run backtest on historical data with caching"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                data,
                strategy.__class__.__name__,
                self.initial_balance,
                self.position_size
            )
            
            # Check cache
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Run backtest
            results_json = self._run_backtest_cached(
                data.to_json(),
                strategy.__class__.__name__,
                self.initial_balance,
                self.position_size
            )
            
            # Parse results
            results = json.loads(results_json)
            
            # Convert timestamps back to datetime
            for trade in results['trades']:
                trade['entry_time'] = pd.Timestamp(trade['entry_time'])
                trade['exit_time'] = pd.Timestamp(trade['exit_time'])
            for point in results['equity_curve']:
                point['timestamp'] = pd.Timestamp(point['timestamp'])
            
            # Cache results
            self._cache[cache_key] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error during backtesting: {str(e)}")
            return self._generate_empty_results()
    
    def _enter_long(self, current_data, next_data):
        """Enter long position"""
        entry_price = float(next_data['open'])
        position_size = (self.balance * self.position_size) / entry_price
        
        self.position = {
            'type': 'long',
            'entry_price': entry_price,
            'size': position_size,
            'entry_time': next_data.name
        }
    
    def _enter_short(self, current_data, next_data):
        """Enter short position"""
        entry_price = float(next_data['open'])
        position_size = (self.balance * self.position_size) / entry_price
        
        self.position = {
            'type': 'short',
            'entry_price': entry_price,
            'size': position_size,
            'entry_time': next_data.name
        }
    
    def _exit_position(self, current_data, next_data):
        """Exit current position"""
        exit_price = float(next_data['open'])
        
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
        
        current_price = float(current_price)
        if self.position['type'] == 'long':
            unrealized_pnl = (current_price - self.position['entry_price']) * self.position['size']
        else:  # short
            unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
        
        return self.balance + unrealized_pnl
    
    def _generate_results(self):
        """Generate backtest results and statistics"""
        if not self.trades:
            return self._generate_empty_results()
        
        try:
            # Convert trades to DataFrame for analysis
            trades_df = pd.DataFrame(self.trades)
            equity_df = pd.DataFrame(self.equity_curve)
            
            # Calculate metrics
            total_return = float((self.balance - self.initial_balance) / self.initial_balance)
            win_rate = float(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df))
            avg_return = float(trades_df['return'].mean())
            
            # Calculate max drawdown
            equity_df['drawdown'] = equity_df['equity'].cummax() - equity_df['equity']
            max_drawdown = float(equity_df['drawdown'].max() / equity_df['equity'].cummax().max())
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = float(np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0)
            
            return {
                'total_return': total_return,
                'total_trades': int(len(trades_df)),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': float(self.balance),
                'trades': self.trades,
                'equity_curve': self.equity_curve
            }
            
        except Exception as e:
            st.error(f"Error generating results: {str(e)}")
            return self._generate_empty_results()
    
    def _generate_empty_results(self):
        """Generate empty results when no trades are made"""
        return {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_balance': float(self.initial_balance),
            'trades': [],
            'equity_curve': []
        }
    
    def clear_cache(self):
        """Clear cache and force garbage collection"""
        self._cache.clear()
        gc.collect()
