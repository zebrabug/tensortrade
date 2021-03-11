import ta

import pandas as pd

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, RUR
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default
from tensortrade.agents import DQNAgent
import matplotlib.pyplot as plt
from finrl.model.models import DRLAgent

df = pd.read_pickle('./data/OHLC_deals_df.pkl')


agent = None
for i in range(0,130000,3000):
    df_USD = df.iloc[i:i+3000,:]
    df_USD.rename(columns = {
        'Time':'date',
        'Open':'open',
        'Close':'close',
        'Low':'low',
        'High':'high',
        'Volume':'volume'
    }, inplace = True)

    df_USD = df_USD[df_USD['close'].notnull()]
    dataset = ta.add_all_ta_features(df_USD, 'open', 'high', 'low', 'close', 'volume', fillna=True)

    price_history = dataset[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data
    dataset.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

    micex = Exchange("MICEX", 
                    service=execute_order, 
                    options=ExchangeOptions(commission = 0.0003, #0.003,
                                            min_trade_size = 1e-6,
                                            max_trade_size = 1e6,
                                            min_trade_price = 1e-8,
                                            max_trade_price= 1e8,
                                            is_live=False)  )(
                                                Stream.source(price_history['close'].tolist(), dtype="float").rename("RUR-USD"))

    portfolio = Portfolio(RUR, [
        Wallet(micex, 0 * USD),
        Wallet(micex, 73000 * RUR),
    ])

    with NameSpace("MICEX"):
        streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

    feed = DataFeed(streams)
    feed.next()

    env = default.create(
        portfolio=portfolio,
        action_scheme="simple",#"managed-risk", simpleBuy
        reward_scheme="simple",#"risk-adjusted",
        feed=feed,
        renderer="screen-log",  # ScreenLogger used with default settings
        window_size=20
    )

    # if agent is None:
    #     agent = DQNAgent(env)
    # else:
    #     agent = DQNAgent(env,policy_network=agent.policy_network)
    # agent.train(n_episodes=1, n_steps=720, render_interval=10)

    #if agent is None:
    agent = DRLAgent(env = env)
    PPO_PARAMS = {
        "n_steps": 1440,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

    trained_ppo = agent.train_model(model=model_ppo, 
                                tb_log_name='ppo',
                                total_timesteps=50000)
    
    pd.DataFrame(portfolio.performance).transpose()[['net_worth']].to_pickle('./tmp/result3_'+str(i)+'.pkl')