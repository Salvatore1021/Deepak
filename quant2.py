import pandas as pd
import numpy as np
import lightgbm as lgb
import importlib

# Dynamically import the repo's main file since python module names usually can't start with numbers
repo_code = importlib.import_module("80lpa")


# ==========================================================
# ML FEATURE ENGINEERING
# ==========================================================

def create_features(df):
    df_list = []
    for name, group in df.groupby('index_name'):
        group = group.copy()

        # Current Day Returns
        group['returns'] = group['close'].pct_change().dropna()


        # Volatility Adjusted Momentum
        vol_21 = group['returns'].rolling(21).std()
        group['sharpe_mom'] = group['returns'].rolling(21).mean() / (vol_21 + 1e-9)

        # Trend / Regime Filters
        group['sma_50'] = group['close'].rolling(window=50).mean()
        group['dist_sma_50'] = (group['close'] / group['sma_50']) - 1

        # RSI
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        group['rsi_14'] = 100 - (100 / rs)

        # Lagged Returns (Pattern Recognition)
        for lag in [1, 3, 5, 10]:
            group[f'ret_lag_{lag}'] = group['returns'].shift(lag)

        # Target shifted by -2 to simulate real-life 1-day execution delay
        group['target_return'] = group['returns'].shift(-2)

        df_list.append(group)

    full_df = pd.concat(df_list)

    # Critical Fix: Only dropna on features. DO NOT dropna on target_return
    # otherwise we lose the ability to predict on the most recent day.
    feature_cols = ['sharpe_mom', 'dist_sma_50', 'rsi_14', 'ret_lag_1', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10']
    full_df = full_df.dropna(subset=feature_cols)
    return full_df


def prepare_ranking_labels(df, target_col='target_return', n_bins=5):
    df = df.copy()

    def qcut_safe(x):
        valid = x.dropna()
        if len(valid) < n_bins:
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x, n_bins, labels=False, duplicates='drop')

    # Rank based on the REALISTIC target (T+2)
    df['rank_label'] = df.groupby('tradedate')[target_col].transform(qcut_safe)
    df['rank_label'] = df['rank_label'].fillna(n_bins // 2).astype(int)
    return df


def train_and_predict(df):
    features = [
        'sharpe_mom', 'dist_sma_50', 'rsi_14',
        'ret_lag_1', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10'
    ]

    df = df.sort_values(['tradedate', 'index_name'])
    df = prepare_ranking_labels(df)

    # Dynamic Validation Split: Automatically uses first 70% of dates for training
    unique_dates = df['tradedate'].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.7)
    split_date = unique_dates[split_idx]

    # Use .copy() to ensure we safely manipulate these slices later
    train = df[df['tradedate'] < split_date].copy()
    test = df[df['tradedate'] >= split_date].copy()

    # Fallback if the dataset is too small
    if train.empty:
        train = df.copy()
        test = df.copy()

    qids_train = train.groupby("tradedate")["tradedate"].count().to_numpy()
    qids_test = test.groupby("tradedate")["tradedate"].count().to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.04,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
        label_gain=[0, 0, 1, 3, 8]
    )

    print("Training LightGBM Ranker Model...")
    model.fit(
        train[features],
        train['rank_label'],
        group=qids_train,
        eval_set=[(test[features], test['rank_label'])],
        eval_group=[qids_test],
        eval_at=[1]
    )

    test['predicted_score'] = model.predict(test[features])


# ==========================================================
# TRANSLATE ML SCORES TO REPO SIGNALS
# ==========================================================

def generate_signals(df):
    # Initialize signal columns
    df['Buy'] = False
    df['Sell'] = False

    # --- PORTFOLIO RULES ---
    TARGET_POSITIONS = 10  # Diversify risk across up to 10 stocks (Limits: 1 to 100)
    SELL_RANK_THRESHOLD = 30  # Only sell if the stock drops below the Top 30

    # Keep track of what we currently own to apply hysteresis (stickiness)
    current_holdings = set()

    # Ensure data is sorted chronologically for accurate state tracking
    df = df.sort_values(['tradedate', 'index_name'])

    for date, group in df.groupby('tradedate'):
        # Stricter Regime Filter: Stock MUST be trading above its 50-day SMA
        valid_candidates = group[group['dist_sma_50'] > 0].copy()

        if valid_candidates.empty:
            # If the market is crashing, sell everything and hold 100% cash
            sell_idx = group[group['index_name'].isin(current_holdings)].index
            df.loc[sell_idx, 'Sell'] = True
            current_holdings.clear()
            continue

        # Rank today's valid candidates based on the ML score (1 is highest score)
        valid_candidates['daily_rank'] = valid_candidates['predicted_score'].rank(ascending=False, method='first')

        # ==========================================
        # 1. PROCESS SELLS FIRST
        # ==========================================
        sells_today = []
        for stock in list(current_holdings):
            # Sell Rule A: The stock dropped below its 50-day SMA (no longer in valid_candidates)
            if stock not in valid_candidates['index_name'].values:
                sells_today.append(stock)
            else:
                # Sell Rule B: The stock's ML rank has deteriorated past our threshold
                stock_rank = valid_candidates.loc[valid_candidates['index_name'] == stock, 'daily_rank'].values[0]
                if stock_rank > SELL_RANK_THRESHOLD:
                    sells_today.append(stock)

        # Execute Sells
        if sells_today:
            # Match the stock names back to the dataframe index for this specific date
            sell_idx = group[group['index_name'].isin(sells_today)].index
            df.loc[sell_idx, 'Sell'] = True
            for stock in sells_today:
                current_holdings.remove(stock)

        # ==========================================
        # 2. PROCESS BUYS
        # ==========================================
        # Calculate how many slots are open in our 10-stock portfolio
        open_slots = TARGET_POSITIONS - len(current_holdings)

        if open_slots > 0:
            # Filter out stocks we already own
            potential_buys = valid_candidates[~valid_candidates['index_name'].isin(current_holdings)]

            # Select the absolute best remaining stocks to fill the empty slots
            top_buys = potential_buys.nsmallest(open_slots, 'daily_rank')

            if not top_buys.empty:
                buy_idx = top_buys.index
                df.loc[buy_idx, 'Buy'] = True
                for stock in top_buys['index_name'].values:
                    current_holdings.add(stock)

    return df


# ==========================================================
# MAIN EXECUTION (USING REPO'S UNMODIFIED BACKTESTER)
# ==========================================================

def main():
    # 1. Load data
    data = pd.read_csv("filled_indices.csv", parse_dates=["tradedate"])

    # CRITICAL FIX: Sort the data so shift() and rolling() don't mix up stocks/dates!
    data.sort_values(['index_name', 'tradedate'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 2. ML Feature Pipeline
    print("Generating Features...")
    df_features = create_features(data)

    # 3. Train & Predict
    df_with_scores = train_and_predict(df_features)

    # 4. Generate 'Buy'/'Sell' booleans
    data_signals = generate_signals(df_with_scores)

    # 5. Route strictly to repo's unmodified compliant backtest logic
    print("\nRunning Team's Official Backtest Engine...")
    equity_curve, trade_log, turnover = repo_code.run_backtest(data_signals)
    metrics, full_df = repo_code.analyze_performance(equity_curve, data)

    # 6. Generate repo's required standard outputs
    equity_curve.to_csv("equity_curve.csv", index=False)
    trade_log.to_csv("trade_log.csv", index=False)
    full_df[["Date", "Drawdown"]].to_csv("drawdown_curve.csv", index=False)
    pd.DataFrame(metrics, index=[0]).to_csv("performance_metrics.csv", index=False)

    rolling_table = pd.DataFrame({
        "Date": full_df["Date"],
        "Roll_1Y": repo_code.compute_rolling_series(full_df, 252),
        "Roll_3Y": repo_code.compute_rolling_series(full_df, 756),
        "Roll_5Y": repo_code.compute_rolling_series(full_df, 1260)
    })
    rolling_table.to_csv("rolling_outperformance.csv", index=False)
    pd.DataFrame({"Turnover": [turnover]}).to_csv("turnover.csv", index=False)

    print("\n===== ML PERFORMANCE METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"\nPortfolio Turnover: {turnover:.4f}")


if __name__ == "__main__":
    main()
