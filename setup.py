from setuptools import setup, find_packages

setup(
    name="forex_trader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.12.0",
        "plotly>=5.13.0",
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "oandapyV20>=0.7.2",
        "ta>=0.10.0",  # Technical Analysis library
        "joblib>=1.2.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "forex_trader=forex_trader.streamlit_app:main",
        ],
    },
)
