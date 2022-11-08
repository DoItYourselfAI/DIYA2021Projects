import os
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier


def load_data(args):
    dfs, symbols = [], []
    dirpath = os.path.join(args.data_dir, 'dow30')
    for filename in os.listdir(dirpath):
        symbols.append(filename[:-4])
        df = pd.read_csv(os.path.join(dirpath, filename), index_col='Date')
        dfs.append(df)
    data = pd.concat([df.Close for df in dfs], axis=1).dropna()
    data.index = pd.to_datetime(data.index)
    data.columns = symbols
    return data


def portfolio_return(args, method='max_sharpe'):
    # constants
    start_date = args.start_test
    lookback = 252  # days
    transaction_cost = args.transaction_cost  # ratio to the trading volume
    initial_cash = args.initial_balance

    data = load_data(args)

    portfolio = None
    tc = transaction_cost
    net_value = initial_cash
    wallet = {}
    net_values = {}

    start_idx = data.index.tolist().index(data[start_date:].index[0]) + 27 # align trading days with RL (DJIANew uses past 25 days as lookback)
    for idx in range(start_idx, len(data)):
        _data = data.iloc[idx - lookback:idx]
        exp_return = mean_historical_return(_data, frequency=lookback)
        cov_matrix = risk_matrix(_data, frequency=lookback)

        frontier = EfficientFrontier(exp_return, cov_matrix)
        # make initial portfolio
        if portfolio is None:
            portfolio = getattr(frontier, method)()
            prices = data.iloc[idx]
            for symbol, weight in portfolio.items():
                price = prices[symbol]
                quantity = net_value * weight // price
                wallet[symbol] = quantity
                cost = price * abs(quantity) * tc
                net_value -= cost

        else:
            prev_prices = data.iloc[idx - 1]
            prices = data.iloc[idx]
            # recalculate current net value
            for symbol, weight in portfolio.items():
                prev_price = prev_prices[symbol]
                new_price = prices[symbol]
                gain = wallet[symbol] * (new_price - prev_price)
                net_value += gain
            # rebalance
            new_portfolio = getattr(frontier, method)()
            for symbol, weight in new_portfolio.items():
                prev_weight = portfolio[symbol]
                price = prices[symbol]
                quantity = net_value * (weight - prev_weight) // price
                if wallet[symbol] < -quantity:
                    quantity = -wallet[symbol]
                    new_portfolio[symbol] = 0
                wallet[symbol] += quantity
                cost = price * abs(quantity) * tc
                net_value -= cost
            portfolio = new_portfolio

        net_values[data.index[idx]] = net_value

    return pd.Series(net_values)
