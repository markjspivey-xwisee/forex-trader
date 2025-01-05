# Forex Trading System

A modular forex trading system with multiple ML models, technical indicators, and backtesting capabilities.

## Project Structure

```
src/
├── data/
│   ├── fetcher.py      # Data fetching from OANDA
│   └── indicators.py   # Technical indicators
├── models/
│   ├── base_model.py   # Abstract base model
│   ├── random_forest.py # RandomForest implementations
│   └── neural_network.py # MLP and LSTM implementations
├── backtesting/
│   └── backtester.py   # Backtesting engine
├── main.py            # Main entry point
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OANDA credentials:
```
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_TYPE=practice
INSTRUMENT=EUR_USD
GRANULARITY=H1
```

## Usage

The system supports three main modes of operation:

### 1. Training Models

Train all available models on historical data:
```bash
python main.py --mode train --days 90
```

### 2. Backtesting

Run a backtest with a specific model:
```bash
python main.py --mode backtest --model random_forest_conservative --days 90 --balance 10000 --position-size 0.1
```

Available models:
- random_forest_conservative
- random_forest_aggressive
- neural_network
- lstm

### 3. Live Trading Signals

Get current trading signals:
```bash
python main.py --mode trade --model random_forest_conservative
```

## Models

### Random Forest
- Conservative: Lower risk, more stringent trading criteria
- Aggressive: Higher risk, more trading opportunities

### Neural Networks
- MLP: Multi-layer perceptron for pattern recognition
- LSTM: Long Short-Term Memory for sequence prediction

## Features

- Multiple ML models with different risk profiles
- Comprehensive technical indicators
- Robust backtesting engine
- Stop-loss and take-profit management
- Performance metrics and visualization
- Modular design for easy extension

## Development

The project is structured to be modular and extensible:

1. Add new models by inheriting from `BaseModel`
2. Add technical indicators in `indicators.py`
3. Modify backtesting parameters in `backtester.py`
4. Extend data sources in `fetcher.py`

## Performance Optimization

To handle token limits during development:

1. Each component is in a separate module for focused discussions
2. Clear separation of concerns for easier maintenance
3. Modular design allows discussing specific parts without loading entire codebase
4. Results and models are saved with timestamps for easy reference
5. Structured logging and error handling for debugging

## Notes

- The system uses OANDA's API for forex data
- Default timeframe is H1 (1-hour candles)
- Models are saved when validation accuracy exceeds 55%
- Results are saved in JSON format with timestamps
