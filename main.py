import argparse
from data_fetcher import OANDADataFetcher
from strategy import MLStrategy
from backtester import Backtester
import json
import os
from datetime import datetime

def save_results(results, filename):
    """Save backtest results to a JSON file"""
    os.makedirs('results', exist_ok=True)
    with open(f'results/{filename}', 'w') as f:
        json.dump(results, f, indent=4, default=str)

def main():
    parser = argparse.ArgumentParser(description='Forex Trading Agent')
    parser.add_argument('--mode', choices=['train', 'backtest', 'trade'], default='backtest',
                      help='Mode of operation')
    parser.add_argument('--days', type=int, default=90,
                      help='Number of days of historical data to use')
    parser.add_argument('--initial_balance', type=float, default=10000,
                      help='Initial balance for backtesting')
    parser.add_argument('--position_size', type=float, default=0.1,
                      help='Position size as fraction of balance')
    
    args = parser.parse_args()
    
    # Initialize components
    fetcher = OANDADataFetcher()
    strategy = MLStrategy()
    backtester = Backtester(
        initial_balance=args.initial_balance,
        position_size=args.position_size
    )
    
    # Fetch and prepare data
    print(f"Fetching {args.days} days of historical data...")
    data = fetcher.fetch_historical_data(days=args.days)
    data_with_features = fetcher.add_features(data)
    
    if args.mode in ['train', 'backtest']:
        # Train the strategy
        print("\nTraining strategy...")
        metrics = strategy.train(data_with_features)
        print("Training metrics:")
        print(f"Training accuracy: {metrics['train_accuracy']:.2f}")
        print(f"Validation accuracy: {metrics['validation_accuracy']:.2f}")
    
    if args.mode in ['backtest', 'trade']:
        # Run backtest
        print("\nRunning backtest...")
        results = backtester.run(data_with_features, strategy)
        
        # Save and display results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(results, f'backtest_results_{timestamp}.json')
        
        print("\nBacktest Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    if args.mode == 'trade':
        # Get latest data point and make prediction
        latest_features = data_with_features[strategy.feature_columns].iloc[-1]
        prediction = strategy.predict(latest_features)
        
        print("\nLatest Trading Signal:")
        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        print(f"Signal: {signal_map[prediction]}")
        print(f"Timestamp: {data_with_features.index[-1]}")
        print(f"Current Price: {data_with_features['close'].iloc[-1]:.5f}")

if __name__ == "__main__":
    main()
