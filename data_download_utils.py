import pandas as pd

def get_query(ticker, period1, period2, interval):
    return f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

def download_dataset(tickers, period1, period2, interval, filename):
    #create dataset with Date column
    ticker = tickers[0]
    query_string = get_query(ticker, period1, period2, interval)
    df_this = pd.read_csv(query_string)
    df_stocks = pd.DataFrame()
    df_stocks["Date"] = df_this["Date"]

    #add all stocks
    for ticker in tickers:
        query_string = get_query(ticker, period1, period2, interval)
        df_this = pd.read_csv(query_string)
        df_stocks[ticker] = df_this["Close"].round(decimals=3)

    #save df_stocks
    df_stocks.to_pickle("Data/" + filename)
