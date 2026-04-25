"""
data_loader.py
--------------
Downloads historical daily adjusted-close prices for a list of tickers
using yfinance and saves them as a single CSV with dates as the index.

Usage (from the terminal, inside your project folder):
    python data_loader.py

Requirements:
    pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import os

# ---------------------------------------------------------------------------
# EDIT THIS LIST to choose your candidate pairs.
# Each inner list is one candidate pair. You need at least 3-5 pairs to screen.
# The screener will keep only the ones that pass the cointegration test.
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS = [
    # US pairs (use these if you want the easiest, most liquid data)
    ("KO",   "PEP"),     # Coca-Cola vs Pepsi
    ("XOM",  "CVX"),     # Exxon Mobil vs Chevron
    ("MA",   "V"),       # Mastercard vs Visa
    ("GLD",  "SLV"),     # Gold ETF vs Silver ETF
    ("HD",   "LOW"),     # Home Depot vs Lowe's

    # Indian pairs (append ".NS" for NSE tickers on yfinance)
    ("HDFCBANK.NS", "ICICIBANK.NS"),
    ("TCS.NS",      "INFY.NS"),
    ("RELIANCE.NS", "ONGC.NS"),
]

# Use 4 years: first 3 for training (pair selection + OU fit), last 1 for backtest.
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"

OUTPUT_FILE = "prices.csv"


def download_prices(tickers, start, end):
    """Download adjusted close prices for a list of tickers."""
    tickers = list(set(tickers))  # de-duplicate
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    # auto_adjust=True makes 'Close' already adjusted for dividends and splits
    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"]
    # yfinance returns a Series if only one ticker; force DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df.dropna(how="all")               # drop rows where all prices missing
    df = df.ffill().dropna()                # forward-fill small gaps, then drop any remaining NaN rows
    return df


def main():
    all_tickers = [t for pair in CANDIDATE_PAIRS for t in pair]
    prices = download_prices(all_tickers, START_DATE, END_DATE)
    prices.to_csv(OUTPUT_FILE)
    print(f"\nSaved {len(prices)} rows, {len(prices.columns)} columns to {OUTPUT_FILE}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print("\nFirst 3 rows:")
    print(prices.head(3).round(2))


if __name__ == "__main__":
    main()
