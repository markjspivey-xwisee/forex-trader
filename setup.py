from setuptools import setup, find_packages

setup(
    name="forex_trader",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.41.1",
        "pandas>=2.2.3",
        "numpy>=2.0.2",
        "plotly>=5.24.1",
        "oandapyV20>=0.7.2",
        "scikit-learn>=1.6.0",
        "tensorflow>=2.18.0",
        "xgboost>=2.1.3",
        "python-dotenv>=1.0.1",
    ],
)
