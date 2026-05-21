import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse as ap

ap = ap.ArgumentParser(description='Calculate option metrics for a given stock symbol.')
ap.add_argument('symbol', help='Stock symbol for which to calculate option metrics.')
ap.add_argument('--expiry', help='Option expiry date in YYYY-MM-DD format. Defaults to the nearest expiry.', default=None)

args = ap.parse_args()
symbol = args.symbol
expiry_str = args.expiry
ticker = yf.Ticker(symbol)
options = ticker.options
price = ticker.history(period="1d")['Close'].iloc[-1]

if not expiry_str:
    expiry_str = options[0]
expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
today = datetime.today()

days_to_expiry = (expiry_date - today).days
T = days_to_expiry / 252 

option_chain = ticker.option_chain(expiry_str)

calls = option_chain.calls.copy()
calls['option_type'] = 'call'
puts = option_chain.puts.copy()
puts['option_type'] = 'put'

option_chain_df = pd.concat([calls, puts], ignore_index=True, sort=False)

# print(option_chain_df.head())

#print(option_chain_df.columns)

usefulstuff = option_chain_df[['contractSymbol', 'option_type', 'lastPrice', 'strike', 'impliedVolatility', 'bid', 'ask']].copy()

usefulstuff.loc[:, 'mid_price'] = (usefulstuff['bid'] + usefulstuff['ask']) / 2
usefulstuff.loc[:, 'diff'] = usefulstuff['strike'] - price

#print(f"Current stock price: {price:.2f}")
atmRow = usefulstuff.sort_values(by='diff', key=abs).iloc[0]

from scipy.stats import norm
import numpy as np
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price

print(f"Fair value of the ATM {atmRow['option_type']} option: {black_scholes(price, atmRow['strike'], T, 0.01, atmRow['impliedVolatility'], atmRow['option_type']):.2f}")


