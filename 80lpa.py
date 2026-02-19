import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

INITIAL_CAPITAL = 5_000_000
TRANSACTION_COST = 0.00268
MIN_HISTORY_DAYS = 100
BENCHMARK = "NSE500"

# ==========================================================
# INDICATORS
# ==========================================================

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_bollinger(close, window=10, n_std=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma + n_std*std, ma - n_std*std

def compute_rsi(close, period=5):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean().replace(0, 1e-9)
    rs = gain / loss
    return 100 - (100/(rs))

def compute_macd(close):
    fast = compute_ema(close, 15)
    slow = compute_ema(close, 35)
    macd = fast - slow
    signal = macd.ewm(span=5, adjust=False).mean()
    return macd, signal

def compute_stochastic(high, low, close):
    lmin = low.rolling(14).min()
    hmax = high.rolling(14).max()
    k = 100*(close - lmin)/(hmax - lmin).replace(0,1e-9)
    d = k.rolling(3).mean()
    return k, d

# ==========================================================
# SIGNALS
# ==========================================================

def generate_signals(df):

    df = df.copy()

    df["EMA20"] = compute_ema(df["close"],20)
    df["EMA100"] = compute_ema(df["close"],100)
    df["BB_up"], df["BB_low"] = compute_bollinger(df["close"])
    df["RSI"] = compute_rsi(df["close"])
    df["MACD"], df["MACD_sig"] = compute_macd(df["close"])
    df["%K"], df["%D"] = compute_stochastic(df["high"], df["low"], df["close"])

    df["Buy"] = (
        ((df["close"] >= df["EMA100"]) |
        (df["EMA20"] >= df["EMA100"])) &
        ((df["close"] <= df["BB_low"]) |
        (df["RSI"] <= 40)) &
        ((df["%K"] <= 20) | (df["%D"] <= 20))
    )

    df["Sell"] = (
        (df["EMA20"] < df["EMA100"]) &
        ((df["close"] >= df["BB_up"])) &
        (df["RSI"] >= 80) &
        (((df["MACD"].shift(1) > df["MACD_sig"].shift(1))) |
        ((df["MACD"] < df["MACD_sig"])))
        
    )

    return df

# ==========================================================
# BACKTEST ENGINE (COMPLIANT EXECUTION)
# ==========================================================
# ==========================================================
# BACKTEST ENGINE (COMPLIANT EXECUTION) - NO CASH DRAG
# ==========================================================

def run_backtest(data):

    cash = INITIAL_CAPITAL
    positions = {}
    equity_curve = []
    trade_log = []
    total_traded_value = 0
    MAX_POSITION_SIZE = .5
    
   
    dates = sorted(data["tradedate"].unique())

    for i in range(MIN_HISTORY_DAYS, len(dates)-1):

        today = dates[i]
        tomorrow = dates[i+1]

        df_today = data[data["tradedate"]==today]
        df_next = data[data["tradedate"]==tomorrow]

        # --- EXIT ---
        for sym in list(positions.keys()):
            row = df_today[df_today["index_name"]==sym]
            if not row.empty and row["Sell"].iloc[0]:

                row_next = df_next[df_next["index_name"]==sym]
                if not row_next.empty:

                    exec_price = row_next[["open","high","low","close"]].mean(axis=1).iloc[0]
                    sell_price = exec_price*(1-TRANSACTION_COST)

                    qty = positions.pop(sym)
                    traded_value = qty*exec_price
                    total_traded_value += traded_value

                    cash += qty*sell_price

                    trade_log.append([tomorrow,sym,"SELL",qty,exec_price])

        # --- ENTRY ---
        buys = df_today[
            df_today["Buy"] &
            ~df_today["index_name"].isin(positions)
        ]

        if not buys.empty:
            
            current_portfolio_value = cash
            for sym, qty in positions.items():
                sym_row = df_today[df_today["index_name"]==sym]
                if not sym_row.empty:
                    current_portfolio_value += qty * sym_row["close"].iloc[0]

            max_allocation = current_portfolio_value * MAX_POSITION_SIZE

            for sym in buys["index_name"]:
                if cash <= 0:
                    break 
                
                allocation = min(max_allocation, cash)

                row_next = df_next[df_next["index_name"]==sym]
                if not row_next.empty:

                    exec_price = row_next[["open","high","low","close"]].mean(axis=1).iloc[0]
                    buy_price = exec_price*(1+TRANSACTION_COST)

                    qty = allocation//buy_price
                    cost = qty*buy_price

                    if qty>0 and cost<=cash:

                        traded_value = qty*exec_price
                        total_traded_value += traded_value

                        positions[sym]=qty
                        cash-=cost

                        trade_log.append([tomorrow,sym,"BUY",qty,exec_price])

        # --- DAILY VALUE ---
        portfolio_value = cash
        for sym,qty in positions.items():
            close_price = df_today[df_today["index_name"]==sym]["close"].iloc[0]
            portfolio_value += qty*close_price

        equity_curve.append({
            "Date":today,
            "PortfolioValue":portfolio_value,
            "Positions":len(positions)
        })

    equity_curve = pd.DataFrame(equity_curve)
    trade_log = pd.DataFrame(trade_log,columns=["Date","Symbol","Side","Qty","ExecPrice"])

    turnover = total_traded_value / INITIAL_CAPITAL

    return equity_curve, trade_log, turnover

#PERFORMANCE MATRICS

def compute_drawdown(series):
    running_max = series.cummax()
    return series/running_max - 1

def compute_cagr(df):
    years = (df["Date"].iloc[-1]-df["Date"].iloc[0]).days/365.25
    return (df["PortfolioValue"].iloc[-1]/INITIAL_CAPITAL)**(1/years)-1

def compute_vol(df):
    return df["Ret"].std()*np.sqrt(252)

def compute_ir(df):
    excess = df["Ret"]-df["BRet"]
    return (excess.mean()*252)/(excess.std()*np.sqrt(252))

def compute_up_down_capture(df):
    up = df[df["BRet"]>0]
    down = df[df["BRet"]<0]
    up_cap = ((1+up["Ret"]).prod()-1)/((1+up["BRet"]).prod()-1)
    down_cap = ((1+down["Ret"]).prod()-1)/((1+down["BRet"]).prod()-1)
    return up_cap, down_cap

def compute_rolling_outperformance(df, window):
    strat = (1+df["Ret"]).rolling(window).apply(np.prod, raw=True)-1
    bench = (1+df["BRet"]).rolling(window).apply(np.prod, raw=True)-1
    outperf = strat-bench
    return outperf.mean(), outperf.min()

def compute_rolling_series(df, window):
    strat = (1 + df["Ret"]).rolling(window).apply(np.prod, raw=True) - 1
    bench = (1 + df["BRet"]).rolling(window).apply(np.prod, raw=True) - 1
    return strat - bench
# ==========================================================
# ANALYSIS PIPELINE
# ==========================================================

def analyze_performance(equity_curve, raw_data):

    bench = raw_data[raw_data["index_name"]==BENCHMARK][
        ["tradedate","close"]
    ].rename(columns={"tradedate":"Date","close":"BenchClose"})

    df = pd.merge(equity_curve,bench,on="Date").dropna()

    df["Ret"] = df["PortfolioValue"].pct_change()
    df["BRet"] = df["BenchClose"].pct_change()
    df = df.dropna()

    df["Drawdown"] = compute_drawdown(df["PortfolioValue"])

    up_cap, down_cap = compute_up_down_capture(df)

    metrics = {
        "Final Value":df["PortfolioValue"].iloc[-1],
        "CAGR":compute_cagr(df),
        "Volatility":compute_vol(df),
        "MaxDD":df["Drawdown"].min(),
        "Information Ratio":compute_ir(df),
        "UpCapture":up_cap,
        "DownCapture":down_cap,
        "1Y Avg Outperformance":compute_rolling_outperformance(df,252)[0],
        "1Y Worst Underperformance":compute_rolling_outperformance(df,252)[1],
        "3Y Avg Outperformance":compute_rolling_outperformance(df,756)[0],
        "3Y Worst Underperformance":compute_rolling_outperformance(df,756)[1],
        "5Y Avg Outperformance":compute_rolling_outperformance(df,1260)[0],
        "5Y Worst Underperformance":compute_rolling_outperformance(df,1260)[1]
    }

    return metrics, df

# ==========================================================
# MAIN
# ==========================================================

def main():

    data = pd.read_csv("filled_indices.csv",parse_dates=["tradedate"])

    processed=[]
    for _,df in data.groupby("index_name"):
        if len(df)>=MIN_HISTORY_DAYS:
            processed.append(generate_signals(df))

    data_signals = pd.concat(processed)

    equity_curve, trade_log, turnover = run_backtest(data_signals)

    metrics, full_df = analyze_performance(equity_curve,data)

    # OUTPUTS REQUIRED
    equity_curve.to_csv("equity_curve.csv",index=False)
    trade_log.to_csv("trade_log.csv",index=False)
    full_df[["Date","Drawdown"]].to_csv("drawdown_curve.csv",index=False)

    # Export metrics
    pd.DataFrame(metrics, index=[0]).to_csv("performance_metrics.csv", index=False)

    # Export rolling table
    rolling_table = pd.DataFrame({
    "Date": full_df["Date"],
    "Roll_1Y": compute_rolling_series(full_df,252),
    "Roll_3Y": compute_rolling_series(full_df,756),
    "Roll_5Y": compute_rolling_series(full_df,1260)
})
    rolling_table.to_csv("rolling_outperformance.csv", index=False)

    # Export turnover
    pd.DataFrame({"Turnover":[turnover]}).to_csv("turnover.csv", index=False)

    print("\n===== PERFORMANCE METRICS =====")
    for k,v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(f"\nPortfolio Turnover: {turnover:.4f}")

if __name__=="__main__":
    main()