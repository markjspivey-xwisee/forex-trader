# AI-Powered Forex Trading Agent

An advanced forex trading system that combines machine learning models with technical analysis for automated trading decisions.

## Features

### Machine Learning Models
- Random Forest (Conservative & Aggressive)
- Neural Networks (MLP & LSTM)
- Model Ensemble capabilities
- Feature importance analysis
- Multiple feature selection methods

### Technical Analysis
- Customizable indicators:
  - Moving Averages (Multiple periods)
  - Bollinger Bands
  - MACD
  - RSI
  - ATR
  - Stochastic Oscillator

### Risk Management
- Dynamic position sizing
- Stop-loss and take-profit optimization
- Maximum trade limits
- Risk/reward analysis
- Portfolio management tools

### Advanced Features
- GPU acceleration support
- Bayesian optimization for hyperparameters
- Ensemble model voting
- Feature selection methods (PCA, RFE)
- Real-time market analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex-trader.git
cd forex-trader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r src/requirements.txt
```

4. Set up your OANDA credentials:
Create a `.env` file in the src directory with:
```
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_TYPE=practice
INSTRUMENT=EUR_USD
GRANULARITY=H1
```

## Usage

### Running the Web Interface
```bash
cd src
streamlit run app.py
```

### Command Line Interface
```bash
# Train models
python main.py --mode train --days 90

# Run backtest
python main.py --mode backtest --model random_forest_conservative --days 90 --balance 10000 --position-size 0.1

# Get live trading signals
python main.py --mode trade --model ensemble
```

## Project Structure

```
src/
├── data/               # Data handling and technical indicators
├── models/            # ML model implementations
├── backtesting/      # Backtesting engine
├── app.py            # Streamlit web interface
├── main.py           # CLI entry point
└── requirements.txt  # Project dependencies
```

## Features in Detail

### Model Types
- **Conservative Random Forest**: Lower risk, higher precision
- **Aggressive Random Forest**: Higher risk, more opportunities
- **Neural Network**: Pattern recognition
- **LSTM**: Sequence prediction
- **Ensemble**: Weighted voting system

### Technical Indicators
- Multiple timeframe analysis
- Customizable indicator parameters
- Visual analysis tools
- Real-time updates

### Risk Management
- Position sizing calculator
- Risk per trade limits
- Maximum drawdown controls
- Portfolio exposure limits

### Advanced Settings
- GPU acceleration
- Feature selection methods
- Hyperparameter optimization
- Model ensemble weights

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
