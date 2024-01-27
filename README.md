# us-market-data

Provides daily U.S. stock market adjusted close data going back to 1885.

data/adjusted_close_data contains adjusted close data that gets updated to the present when the Jupyter Notebook or Python script are ran.

data/full_data.csv contains data until June 16, 2023 with additional columns used to calculate the final adjusted close.

Dividends are added to price-index only data to approximate SPY as closely as possible. As such, by default, a 0.0945% expense ratio is applied throughout (SPY expense ratio as of January 27, 2024). Additionally, SPY adjusted close data is used to fill in dates since June 16, 2023. 

---

## Getting Started

### Through Jupyter Lab (recommended):

Windows:
```shell
git clone https://github.com/SteelCerberus/us-market-data.git
pip install jupyterlab
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user
jupyter lab
```

Linux/Unix:
```shell
git clone https://github.com/SteelCerberus/us-market-data.git
pip install jupyterlab
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user
jupyter lab
```

### Through the Python script:
Windows:
```shell
git clone https://github.com/SteelCerberus/us-market-data.git
pip install jupyterlab
python -m venv venv
venv\Scripts\Activate
python src/market_data.py
```

Linux/Unix:
```shell
git clone https://github.com/SteelCerberus/us-market-data.git
pip install jupyterlab
python -m venv venv
source venv/bin/activate
python src/market_data.py
```

---

## Stock Close Data Sources:

### March 20, 1885 to December 30, 1927:

Dow Jones composite portfolio (industrial and railroad stocks) with dividends added

Dow Jones data source: https://www.billschwert.com/dstock.htm

Dividend data source: http://www.econ.yale.edu/~shiller/data.htm 

### January 3, 1928 to June 15, 1962:

S&P 500 price-only index data with divdends added

S&P 500 data source: https://www.billschwert.com/dstock.htm

Dividend data source: http://www.econ.yale.edu/~shiller/data.htm 

### June 18, 1962 to June 16, 2023:

S&P 500 price-only index data with divdends added

S&P 500 data source: https://finance.yahoo.com/quote/%5EGSPC/history

Dividend data source: http://www.econ.yale.edu/~shiller/data.htm 

### June 20, 2023 to Present:

SPY ETF

SPY data source: https://finance.yahoo.com/quote/SPY/history?p=SPY
