# This file is part of NBA Player Predictor.
# Copyright (C) 2025 John LaVergne
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY.
# See the GNU General Public License for more details.
# <https://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from constants import ROLLING_TREND_WINDOW, ROLLING_WINDOW, TARGET_COLUMN, FEATURE_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes preprocessed DataFrame (with team stats merged) and adds rolling, trend,
    efficiency and contextual features.
    """
    out = df.sort_values(['Player', 'Date']).copy()
    
    for col in ['PTS','FGA','FTA','TOV','DRtg','Pace',
                'eFG%_y','TOV%','DRB%']:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    
    last_played = out.groupby('Player')['Date'].shift(1)
    out['Days_of_rest'] = (out['Date'] - last_played).dt.days.fillna(0)
    out = out[out['MP'].notna() & (out['MP'] != '') & (out["MP"] != 'Inactive') & (out["MP"] != 'Did Not Play') & (out["MP"] != 'Did Not Dress') & (out["MP"] != 'Not With Team')]
    # out["MP"] = pd.to_numeric(out["MP"], errors='coerce').fillna(0)
    
    out['month']     = out['Date'].dt.month
    out['month_sin'] = np.sin(2*np.pi*out['month']/12)
    out['month_cos'] = np.cos(2*np.pi*out['month']/12)
    out.drop(columns=['month'], inplace=True)
    
    out['is_back2back'] = (out['Days_of_rest'] == 1).astype(int)
    # Rolling aggregates
    out['PTS_last_5_avg'] = out['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).mean()
    out['MP_last_5_avg']  = out['MP'].rolling(ROLLING_TREND_WINDOW, min_periods=1).mean()

    # Trend (slope) helper
    def _slope(x):
        if len(x) >= 2:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return 0

    out['PTS_trend_5'] = out['PTS_last_5_avg'].rolling(ROLLING_TREND_WINDOW, min_periods=1).apply(_slope, raw=False)

    # Volatility
    out['PTS_vol_5'] = out['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).std().fillna(0)

    # Efficiency: PTS per minute (5-game ewm)
    raw_eff = out['PTS'] / out['MP'].replace(0, np.nan)
    out['PTS_per_min'] = raw_eff.ewm(span=ROLLING_TREND_WINDOW, adjust=False).mean().fillna(0)

    # Opponent adjustment: league avg DRtg vs opp
    league_avg = out['DRtg'].mean()
    out['def_adj'] = out['PTS_last_5_avg'] * (league_avg / out['DRtg'])
    out['usage_rate'] = (out['FGA'] + 0.44*out['FTA'] + out['TOV']) / out['MP']
        
    return out


def get_train_test_splits(feat_df: pd.DataFrame):
    """
    From feature-engineered DataFrame, split into X, X_test, X_train, y_train, y_test
    using a fixed 80/20 split.
    """
    # Define feature columns
    
    y = feat_df[TARGET_COLUMN]
    X = feat_df[FEATURE_COLUMNS]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prep_game(
    opponent: str,
    raw_player_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    home
) -> pd.DataFrame:
    def safe_polyfit(window: np.ndarray) -> float:
        if len(window) < 2:
            return 0.0
        try:
            return np.polyfit(np.arange(len(window)), window, 1)[0]
        except np.linalg.LinAlgError:
            return 0.0
    """
    Build the feature‐vector for a player's NEXT game vs `opponent`.
    We:
      1. Preprocess and merge historical logs.
      2. Engineer rolling/contextual features on history.
      3. Pull off the last N games to compute “next‐game” statistics.
      4. Fetch opponent stats and assemble a new 1×len(FEATURE_COLUMNS) row.
    """
    is_home_game = 1 if home == "Home" else 0

  
    # 1) Grab the last 10 real games (or fewer if <10 exist)
    last_n = raw_player_df.tail(10)
    
    # Compute “next game”'s rolling features from last_n
    pts_avg   = last_n['PTS'].rolling(ROLLING_WINDOW, min_periods=1).mean().iloc[-1]
    mp_avg    = last_n['MP'].rolling(ROLLING_WINDOW, min_periods=1).mean().iloc[-1]
    
    
    trend = (
        last_n['PTS_last_5_avg']
        .rolling(ROLLING_TREND_WINDOW, min_periods=1)
        .apply(safe_polyfit, raw=False)
        .iloc[-1]
    )
    volatility     = last_n['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).std().shift(1).fillna(0).iloc[-1]
    # ewm efficiency
    raw_eff   = last_n['PTS'] / last_n['MP'].replace(0, 1e-6)
    eff_ewm   = raw_eff.ewm(span=ROLLING_WINDOW, adjust=False).mean().iloc[-1]
    # days of rest
    played    = raw_player_df.loc[raw_player_df['MP']>0, 'Date']
    days_rest = (raw_player_df['Date'].max() - played.iloc[-2]).days if len(played) >= 2 else 0

    # Opponent stats
    opp = team_stats_df.loc[team_stats_df['Team']==opponent].iloc[0]
    league_avg = team_stats_df['DRtg'].mean()
    def_adj    = pts_avg * (league_avg / opp['DRtg'])
    
    fga_sum = last_n['FGA'].sum()
    fta_sum = last_n['FTA'].sum()
    tov_sum = last_n['TOV'].sum()
    mp_sum  = last_n['MP'].sum() or 1e-6
    usage_rate = (fga_sum + 0.44 * fta_sum + tov_sum) / mp_sum
    
    last_month = int(last_n['Date'].iloc[-1].month)
    month_sin  = np.sin(2 * np.pi * last_month / 12)
    month_cos  = np.cos(2 * np.pi * last_month / 12)
    
    last_game_date = raw_player_df['Date'].max()
    today = pd.Timestamp.today().normalize()
    is_back2back = 1 if (today - last_game_date).days == 1 else 0

    # Build the one-row DataFrame, using whatever features you like
    vals = {
        "Home":           is_home_game,
        "Pace":           opp["Pace"],
        "eFG%_y":         opp["eFG%"],
        "TOV%":           opp["TOV%"],
        "DRB%":           opp["DRB%"],
        "PTS_last_5_avg": pts_avg,
        "MP_last_5_avg":  mp_avg,
        "PTS_trend_5":    trend,
        "PTS_vol_5":      volatility,
        "PTS_per_min":    eff_ewm,
        "def_adj":        def_adj,
        "Days_of_rest":   days_rest,
        "usage_rate": usage_rate,
        "month_sin" : month_sin,
        "month_cos" : month_cos,
        "is_back2back": is_back2back,
    }
    
    # filter to only the features being utilized
    row = {col: vals[col] for col in FEATURE_COLUMNS}
    # 5) Return as a 1×N DataFrame in the exact order the model expects
    return pd.DataFrame([row])[FEATURE_COLUMNS]