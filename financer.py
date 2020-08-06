import yfinance as yt
import pandas as pd


def get_ticker(stock: str, start_date = '2000-01-01', requested_columns = ['Open', 'Close']) -> pd.DataFrame:
    ticker = yt.Ticker(stock)
    df = ticker.history(start=start_date)
    return df[requested_columns]



