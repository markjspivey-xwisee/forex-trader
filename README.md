# AI-Powered Forex Trading Agent

An AI-powered forex trading agent built with Streamlit, featuring real-time market analysis, technical indicators, and machine learning models.

## Features

- Real-time forex data fetching
- Technical indicators calculation
- Multiple ML models (Random Forest, Neural Network)
- Backtesting capabilities
- Interactive charts and visualizations
- Memory-optimized data processing
- Caching system for better performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex_trader.git
cd forex_trader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```env
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_ID=your_account_id
```

## Local Development

Run the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

1. Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Click "New app" and select your repository

4. Set the following:
   - Main file path: `streamlit_app.py`
   - Python version: 3.9+
   - Add your environment variables in the Secrets management

## Project Structure

```
forex_trader/
├── data/
│   ├── fetcher.py      # Data fetching and caching
│   └── indicators.py   # Technical indicators
├── models/
│   ├── base_model.py   # Base model class
│   ├── random_forest.py
│   └── neural_network.py
├── backtesting/
│   └── backtester.py   # Backtesting engine
├── .streamlit/
│   └── config.toml     # Streamlit configuration
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
└── streamlit_app.py   # Main application
```

## Memory Management

The application includes several optimizations for handling large datasets:

1. Chunked Processing:
   - Data is processed in configurable chunks
   - Automatic garbage collection
   - Memory monitoring

2. Caching System:
   - Multi-level caching
   - TTL settings
   - Size limits
   - Cache clearing functionality

3. Data Validation:
   - Array length checks
   - Broadcasting fixes
   - Type conversion
   - Error handling

## Configuration

The application can be configured through:

1. `.streamlit/config.toml`:
   - Server settings
   - Theme configuration
   - Performance options

2. Environment variables:
   - API credentials
   - Server configuration
   - Debug settings

## Troubleshooting

1. Memory Issues:
   - Adjust chunk size in the sidebar
   - Clear cache using the button
   - Monitor memory usage in System Status

2. Performance:
   - Reduce data range
   - Adjust cache settings
   - Use the optimized configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
