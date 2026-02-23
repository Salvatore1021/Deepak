import pandas as pd
import numpy as np
import lightgbm as lgb
import importlib

# Dynamically import the repo's main file since python module names usually can't start with numbers
try:
    repo_code = importlib.import_module("80lpa")
except ImportError:
    pass


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

    # Predict on the full dataframe so we have scores for all data
    df['predicted_score'] = model.predict(df[features])

    return df


# ==========================================================
# TRANSLATE ML SCORES TO SIGMOID VALUES FOR ENSEMBLE
# ==========================================================

def generate_signals(df):
    """
    Transforms the raw LightGBM ranking scores into bounded sigmoid probabilities.
    Returns the dataframe with the new 'sigmoid_score' column.
    """
    df = df.copy()

    # Apply Sigmoid function: 1 / (1 + e^-x)
    df['sigmoid_score'] = 1 / (1 + np.exp(-df['predicted_score']))

    # Ensure data is sorted for easy merging later
    df = df.sort_values(['tradedate', 'index_name']).reset_index(drop=True)

    return df


# ==========================================================
# MAIN EXECUTION
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

    # 4. Generate Sigmoid Scores
    print("Calculating Sigmoid Scores for Ensemble...")
    data_with_sigmoid = generate_signals(df_with_scores)

    # 5. Save the output so your ensemble script can load it
    output_filename = "model_1_sigmoid_scores.csv"

    # Extracting just the necessary columns to keep the file lightweight for the ensemble
    ensemble_df = data_with_sigmoid[
        ['tradedate', 'index_name', 'close', 'predicted_score', 'sigmoid_score', 'dist_sma_50']]
    ensemble_df.to_csv(output_filename, index=False)

    print(f"\nSuccess! Sigmoid scores saved to {output_filename}.")
    print(ensemble_df)

    return data_with_sigmoid

if __name__ == "__main__":
    df_scores = main()
