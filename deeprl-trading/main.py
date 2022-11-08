import os, glob
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import agents


def train(args):
    agent = getattr(agents, args.agent)(args)
    path = agent.logger.log_dir
    max_eval = float('-inf')

    # save current config to log directory
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, default=str)

    for idx in range(args.train_iter):
        # train agent
        if idx % args.eval_every == 0:
            agent.eval()
        agent.train()

        # save model
        if idx % args.save_step == 0:
            torch.save(
                agent.model.state_dict(),
                os.path.join(path, 'model.pt')
            )
        
        # save best model after burnin-period
        if idx > args.save_max_after:
            if agent.eval_score > max_eval:
                for filename in glob.glob(os.path.join(path, 'best_model*')):
                    os.remove(filename)
                torch.save(
                agent.model.state_dict(),
                os.path.join(path, f'best_model_{idx}.pt')
                )
                max_eval = agent.eval_score



def test(args):
    agent = getattr(agents, args.agent)(args)
    _, ckpt_name = os.path.split(args.checkpoint)
    path = os.path.join(agent.logger.log_dir, '{}.csv'.format(ckpt_name))
    path_returns = os.path.join(agent.logger.log_dir, '{}_returns.csv'.format(ckpt_name))

    # test run
    pnl, returns = agent.test()
    
    # pnl
    # match datetime index from the environment
    agent.env.test()
    index = agent.env.prices.index[-len(pnl):]
    # save series to csv
    pd.Series(pnl, index=index).to_csv(path)

    # returns
    # match datetime index from the environment
    index = agent.env.prices.index[-len(pnl):]
    # save series to csv
    pd.Series(returns, index=index).to_csv(path_returns)


def visualize(args):
    assert os.path.isdir(args.checkpoint), "must provide directory by the checkpoint argument"

    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpt import portfolio_return 

    sns.set_style('whitegrid')

    # load test data
    dfs = []
    for filename in os.listdir(args.checkpoint):
        if filename.endswith('.csv'):
            path = os.path.join(args.checkpoint, filename)
            df = pd.read_csv(path, index_col='Date')
            df.columns = [filename[:-4]]
            dfs.append(df)
    dfs = pd.concat(dfs, axis=1)

    # add DJIA
    djia = pd.read_csv('data/^DJI.csv', index_col='Date').Close
    djia = djia[dfs.index]
    djia = djia / djia.iloc[0] - 1.0
    dfs['DJIA'] = djia

    benchmark = portfolio_return(args, method='min_volatility')
    benchmark = benchmark[dfs.index]
    benchmark = benchmark / benchmark.iloc[0] - 1.0
    dfs['min_volatility'] = benchmark

    dfs.plot(figsize=(20, 10))
    plt.legend(loc='upper left')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/result.png')


def average_seeds(args):
    assert os.path.isdir(args.checkpoint), "must provide directory by the checkpoint argument"

    from datetime import datetime

    # load test data
    dfs = []
    for filename in os.listdir(args.checkpoint):
        if filename.endswith('_returns.csv'):
            path = os.path.join(args.checkpoint, filename)
            df = pd.read_csv(path, index_col='Date')
            df.columns = [filename[:-4]]
            dfs.append(df)
    
    assert len(dfs) == 3 , "Three '_returns.csv' files have to be provided in the checkpoint folder."
    df_avg = pd.concat(dfs, axis=1).mean(axis=1)

    # save average returns
    path = os.path.join(args.checkpoint, "avg_returns", datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, '{}_returns.csv'.format(args.tag))
    df_avg.to_csv(path)


def performance(args):
    assert os.path.isdir(args.checkpoint), "must provide directory by the checkpoint argument"

    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpt import portfolio_return 
    import quantstats as qs
    from datetime import timedelta
    import dataframe_image as dfi
    from datetime import datetime

    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=2.0) # set fontsize scaling for labels, axis, legend, automatically moves legend

    # load test data
    dfs = []
    for filename in os.listdir(args.checkpoint):
        if filename.endswith('.csv'):
            path = os.path.join(args.checkpoint, filename)
            df = pd.read_csv(path, index_col='Date')
            df.columns = [filename[:-4]]
            dfs.append(df)
    dfs = pd.concat(dfs, axis=1)

    # add one day at the start for returns of DJIA and MPT to be of same length
    dfs.index = pd.to_datetime(dfs.index)
    dfs_idx = dfs.index.copy()
    dfs_idx = dfs_idx.insert(0, dfs_idx[0] - timedelta(days=1))
   
    # add DJIA prices
    djia = pd.read_csv('data/^DJI.csv', index_col='Date').Close
    djia.index = pd.to_datetime(djia.index)
    djia = djia[dfs_idx]
    # calculate returns
    djia = djia.iloc[1:] / djia.iloc[:-1].values - 1.0 
    dfs['DJIA'] = djia

    benchmark = portfolio_return(args, method='min_volatility')
    benchmark = pd.concat([pd.Series(args.initial_balance, index=[dfs_idx[0]]), benchmark]) # concat initial balance in first row
    assert (benchmark.index == dfs_idx).all(), "Index of 'dfs' and 'benchmark' are not aligned"
    benchmark = benchmark.iloc[1:] / benchmark.iloc[:-1].values - 1.0
    dfs['Min Volatility'] = benchmark

    # plot equity line
    dfs_cumprod = (1. + dfs).cumprod()
    dfs_cumprod.plot(figsize=(15, 10), alpha=1.0, linewidth=2.5, ylabel="Portfolio Value")
    plt.tight_layout() # remove whitespace around plot
    path = os.path.join("results", args.tag, datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "plot.png"))

    # calculate key performance statistics
    perfstats = []
    periods = 252 # assuming we have daily (excess) returns (as rf assumed 0.)

    cumr = qs.stats.comp(dfs).rename("Cumulative Return")
    cumr = cumr.apply(lambda x: f'{x: .2%}')
    cagr = qs.stats.cagr(dfs).rename("CAGR") # assumes rf = 0
    cagr = cagr.apply(lambda x: f'{x: .2%}')
    vol = qs.stats.volatility(dfs, periods=periods).rename("Volatility (ann.)") # daily to annualized vol.
    vol = vol.apply(lambda x: f'{x: .2%}')
    sharpe = qs.stats.sharpe(dfs).rename("Sharpe Ratio")
    sharpe = sharpe.apply(lambda x: f'{x: .2f}')
    maxdd = qs.stats.max_drawdown(dfs).rename("Max Drawdown")
    maxdd = maxdd.apply(lambda x: f'{x: .2%}')
    dd_days = []
    for columname in dfs.columns:
        dd_days.append(qs.stats.drawdown_details(qs.stats.to_drawdown_series(dfs))[columname].sort_values(by="max drawdown", ascending=True)["days"].iloc[0])
    dd_days = pd.Series(dd_days, index=dfs.columns, name="Max DD days")
    dd_days = dd_days.apply(lambda x: f'{x: .0f}')
    calmar = qs.stats.calmar(dfs).rename("Calmar Ratio")
    calmar = calmar.apply(lambda x: f'{x: .2f}')
    skew = qs.stats.skew(dfs).rename("Skewness")
    skew = skew.apply(lambda x: f'{x: .2f}')
    kurt = qs.stats.kurtosis(dfs).rename("Kurtosis")
    kurt = kurt.apply(lambda x: f'{x: .2f}')

    # calculate alpha, betas
    X = np.array([np.ones_like(dfs.DJIA), dfs.DJIA]).T
    alphabeta = np.linalg.inv(X.T@X)@X.T@dfs
    alpha = alphabeta.iloc[0, :] * periods
    beta = alphabeta.iloc[1, :]
    alpha = alpha.rename("Alpha")
    beta = beta.rename("Beta")
    alpha = alpha.apply(lambda x: f'{x: .2f}')
    beta = beta.apply(lambda x: f'{x: .2f}')

    #append more stats first here...

    perfstats += [cumr, cagr, vol, sharpe, maxdd, dd_days, calmar, skew, kurt, alpha, beta] # then here.
        
    # save results
    perfstats = pd.concat(perfstats, axis=1)
    perfstats.to_csv(os.path.join(path, "perfstats.csv"))    
    dfi.export(perfstats, os.path.join(path, "perfstats.png"))

    # export latex code for table
    with open(os.path.join(path, "latex.txt"), "w") as text_file:
        text_file.write(perfstats.to_latex(float_format="%.2f"))
        text_file.write("\n")
        text_file.write("% Same table transposed:\n")
        text_file.write("\n")
        text_file.write(perfstats.T.to_latex(float_format="%.2f"))


def test_logger(args):
    from utils.logger import Logger

    # initialize logger
    logger = Logger('test', args=args)
    logger.log("Testing logger functionality...")
    logger.log("Logs saved to {}".format(logger.log_dir))

    # log built-in levels
    logger.log("Logging DEBUG level", lvl='DEBUG')
    logger.log("Logging INFO level", lvl='INFO')
    logger.log("Logging WARNING level", lvl='WARNING')
    logger.log("Logging ERROR level", lvl='ERROR')
    logger.log("Logging CRITICAL level", lvl='CRITICAL')

    # add logging level
    logger.add_level('NEW', 21, color='grey')
    logger.log("Logging NEW level", lvl='NEW')

    # check excepthook
    raise Exception("Checking system excepthook")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning applied to Stock Trading - Data Augmentation using Generative Adversarial Networks"
    )
    common = parser.add_argument_group("common configurations")
    common.add_argument("mode", type=str, default='test_logger')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)
    common.add_argument("--config", type=str, default=None)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument("--log_step", type=int, default=10000)
    log.add_argument("--save_step", type=int, default=1000000)
    log.add_argument("--save_max_after", type=int, default=500000)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--data_dir", type=str, default='data')
    dirs.add_argument("--checkpoint", type=str, default=None)

    env = parser.add_argument_group("environment configurations")
    env.add_argument("--env", type=str.lower, default='djia_new')
    env.add_argument("--start_train", type=str, default="2009-01-01")
    env.add_argument("--start_val", type=str, default="2018-01-01")
    env.add_argument("--start_test", type=str, default="2019-12-01")
    env.add_argument("--initial_balance", type=float, default=1e6)
    env.add_argument("--transaction_cost", type=float, default=1e-3)

    training = parser.add_argument_group("training configurations")
    training.add_argument("--agent", type=str.lower, default='ddpg')
    training.add_argument("--arch", type=str.lower, default='cnn',
                          choices=['cnn', 'transformer'])
    training.add_argument("--train_iter", type=int, default=100000000)
    training.add_argument("--eval_every", type=int, default=10000)
    training.add_argument("--update_every", type=int, default=128)
    training.add_argument("--update_epoch", type=int, default=4)
    training.add_argument("--buffer_size", type=int, default=50000)
    training.add_argument("--warmup", type=int, default=1000)
    training.add_argument("--batch_size", type=int, default=32)

    training.add_argument("--lr_critic", type=float, default=1e-4)
    training.add_argument('-lr', "--lr_actor", type=float, default=1e-4)
    training.add_argument("--grad_clip", type=float, default=0.5)
    training.add_argument("--sigma", type=float, default=0.1)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--lambda", type=float, default=0.95)
    training.add_argument("--polyak", type=float, default=0.99)
    training.add_argument("--cr_coef", type=float, default=0.5)
    training.add_argument("--ent_coef", type=float, default=0.0)
    training.add_argument("--cliprange", type=float, default=0.1)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            args.__dict__ = json.load(f)

    # set random seed
    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # use cuda when available
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set logging level
    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    globals()[args.mode](args)
