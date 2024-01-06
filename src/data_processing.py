import yfinance as yf
import pandas as pd

def process_data():
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.plot.line(y="Close", use_index=True)
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    sp500["Tommorow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tommorow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500