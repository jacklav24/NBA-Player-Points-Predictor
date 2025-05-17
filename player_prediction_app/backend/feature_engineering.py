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


# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Takes preprocessed DataFrame (with team stats merged) and adds rolling, trend,
#     efficiency and contextual features.
#     """
#     out = df.sort_values(['Player', 'Date']).copy()
    
#     for col in ['PTS','FGA','FTA','TOV','DRtg','Pace',
#                 'eFG%_y','TOV%','DRB%']:
#         out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    
#     last_played = out.groupby('Player')['Date'].shift(1)
#     out['Days_of_rest'] = (out['Date'] - last_played).dt.days.fillna(0)
#     out = out[out['MP'].notna() & (out['MP'] != '') & (out["MP"] != 'Inactive') & (out["MP"] != 'Did Not Play') & (out["MP"] != 'Did Not Dress') & (out["MP"] != 'Not With Team')]
#     # out["MP"] = pd.to_numeric(out["MP"], errors='coerce').fillna(0)
    
#     out['month']     = out['Date'].dt.month
#     out['month_sin'] = np.sin(2*np.pi*out['month']/12)
#     out['month_cos'] = np.cos(2*np.pi*out['month']/12)
#     out.drop(columns=['month'], inplace=True)
    
#     out['is_back2back'] = (out['Days_of_rest'] == 1).astype(int)
#     # Rolling aggregates
#     out['PTS_last_5_avg'] = out['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).mean()
#     out['MP_last_5_avg']  = out['MP'].rolling(ROLLING_TREND_WINDOW, min_periods=1).mean()

#     # Trend (slope) helper
#     def _slope(x):
#         if len(x) >= 2:
#             return np.polyfit(np.arange(len(x)), x, 1)[0]
#         return 0

#     out['PTS_trend_5'] = out['PTS_last_5_avg'].rolling(ROLLING_TREND_WINDOW, min_periods=1).apply(_slope, raw=False)

#     # Volatility
#     out['PTS_vol_5'] = out['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).std().fillna(0)

#     # Efficiency: PTS per minute (5-game ewm)
#     raw_eff = out['PTS'] / out['MP'].replace(0, np.nan)
#     out['PTS_per_min'] = raw_eff.ewm(span=ROLLING_TREND_WINDOW, adjust=False).mean().fillna(0)


#     # Opponent adjustment: league avg DRtg vs opp
#     league_avg = out['DRtg'].mean()
#     out['def_adj'] = out['PTS_last_5_avg'] * (league_avg / out['DRtg'])
#     out['usage_rate'] = (out['FGA'] + 0.44*out['FTA'] + out['TOV']) / out['MP']
        
#     return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes preprocessed DataFrame (with team stats merged) and adds:
    - Rolling averages and volatility (3, 5, 10)
    - Opponent-adjusted stats
    - Trend and hot streak flags
    - Z-score based performance normalization
    - Contextual features like rest and calendar encoding
    """
    out = df.sort_values(['Player_ID', 'Date']).copy()

    # Ensure numeric types for key stats
    for col in ['PTS', 'FGA', 'FTA', '3PA', 'TOV', 'DRtg', 'Pace', 'eFG%_y', 'TOV%', 'DRB%']:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    # --- Rest Days & Game Context ---
    last_played = out.groupby('Player_ID')['Date'].shift(1)
    out['Days_of_rest'] = (out['Date'] - last_played).dt.days.fillna(0)
    out['is_back2back'] = (out['Days_of_rest'] == 1).astype(int)

    # Remove DNPs and convert MP to numeric
    out = out[out['MP'].notna() & (out['MP'] != '') & (~out['MP'].isin(['Inactive', 'Did Not Play', 'Did Not Dress', 'Not With Team']))]
    out['MP'] = pd.to_numeric(out['MP'], errors='coerce').fillna(0)

    # --- Date Encoding ---
    out['month']     = out['Date'].dt.month
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out.drop(columns=['month'], inplace=True)

    # --- Usage Rate (needs MP cleaned first) ---
    out['usage_rate'] = (out['FGA'] + 0.44 * out['FTA'] + out['TOV']) / out['MP'].replace(0, np.nan)

    # --- Rolling Features ---
    ROLLING_WINDOWS = [3, 5, 10]
    ROLLING_VARS = ['PTS', 'FGA', '3PA', 'FTA', 'TOV', 'MP', 'usage_rate']

    for var in ROLLING_VARS:
        for w in ROLLING_WINDOWS:
            out[f'{var}_r{w}'] = (
                out.groupby('Player_ID')[var]
                   .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
            )

    # --- Rolling Std (volatility) ---
    for var in ['PTS', 'FGA']:
        out[f'{var}_std_r5'] = (
            out.groupby('Player_ID')[var]
               .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
               .fillna(0)
        )

    # --- Interaction Terms ---
    out['Opp_DRtg_x_PTSr5'] = out['DRtg'] * out['PTS_r5']
    out['Opp_Pace_x_FGAr5'] = out['Pace'] * out['FGA_r5']
    out['Opp_eFG_x_PTSr5']  = out['eFG%_y'] * out['PTS_r5']

    # --- Trend (Slope of rolling PTS) ---
    def _slope(x):
        if len(x) >= 2:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return 0

    out['PTS_trend_5'] = (
        out.groupby('Player_ID')['PTS_r5']
           .transform(lambda x: x.rolling(window=5, min_periods=2).apply(_slope, raw=False))
           .fillna(0)
    )

    # --- EWMA PTS per minute (efficiency) ---
    raw_eff = out['PTS'] / out['MP'].replace(0, np.nan)
    out['PTS_per_min'] = (
        raw_eff.ewm(span=5, adjust=False).mean().fillna(0)
    )

    # --- Hot Streak Flag ---
    rolling_mean = out.groupby("Player_ID")['PTS'].transform(lambda x: x.expanding().mean().shift(1))
    out['Hot_Streak'] = (
        (out['PTS'] > rolling_mean)
        .astype(int)
        .groupby(out['Player_ID'])
        .transform(lambda x: x.rolling(window=3, min_periods=1).sum())
        .fillna(0)
    )

    # --- Z-Score Normalization of PTS vs Player History ---
    expanding_mean = out.groupby('Player_ID')['PTS'].transform(lambda x: x.expanding().mean())
    expanding_std  = out.groupby('Player_ID')['PTS'].transform(lambda x: x.expanding().std()).replace(0, 1)
    out['PTS_z'] = (out['PTS'] - expanding_mean) / expanding_std

    # --- Opponent Adjusted Scoring ---
    league_avg_drtg = out['DRtg'].mean()
    out['def_adj'] = out['PTS_r5'] * (league_avg_drtg / out['DRtg'].replace(0, np.nan))

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


# def prep_game(
#     opponent: str,
#     raw_player_df: pd.DataFrame,
#     team_stats_df: pd.DataFrame,
#     home
# ) -> pd.DataFrame:
#     def safe_polyfit(window: np.ndarray) -> float:
#         if len(window) < 2:
#             return 0.0
#         try:
#             return np.polyfit(np.arange(len(window)), window, 1)[0]
#         except np.linalg.LinAlgError:
#             return 0.0
#     """
#     Build the feature‐vector for a player's NEXT game vs `opponent`.
#     We:
#       1. Preprocess and merge historical logs.
#       2. Engineer rolling/contextual features on history.
#       3. Pull off the last N games to compute “next‐game” statistics.
#       4. Fetch opponent stats and assemble a new 1×len(FEATURE_COLUMNS) row.
#     """
#     is_home_game = 1 if home == "Home" else 0

  
#     # 1) Grab the last 10 real games (or fewer if <10 exist)
#     last_n = raw_player_df.tail(10)
    
#     # Compute “next game”'s rolling features from last_n
#     pts_avg   = last_n['PTS'].rolling(ROLLING_WINDOW, min_periods=1).mean().iloc[-1]
#     mp_avg    = last_n['MP'].rolling(ROLLING_WINDOW, min_periods=1).mean().iloc[-1]
    
    
#     trend = (
#         last_n['PTS_last_5_avg']
#         .rolling(ROLLING_TREND_WINDOW, min_periods=1)
#         .apply(safe_polyfit, raw=False)
#         .iloc[-1]
#     )
#     volatility     = last_n['PTS'].rolling(ROLLING_TREND_WINDOW, min_periods=1).std().shift(1).fillna(0).iloc[-1]
#     # ewm efficiency
#     raw_eff   = last_n['PTS'] / last_n['MP'].replace(0, 1e-6)
#     eff_ewm   = raw_eff.ewm(span=ROLLING_WINDOW, adjust=False).mean().iloc[-1]
#     # days of rest
#     played    = raw_player_df.loc[raw_player_df['MP']>0, 'Date']
#     days_rest = (raw_player_df['Date'].max() - played.iloc[-2]).days if len(played) >= 2 else 0

#     # Opponent stats
#     opp = team_stats_df.loc[team_stats_df['Team']==opponent].iloc[0]
#     league_avg = team_stats_df['DRtg'].mean()
#     def_adj    = pts_avg * (league_avg / opp['DRtg'])
    
#     fga_sum = last_n['FGA'].sum()
#     fta_sum = last_n['FTA'].sum()
#     tov_sum = last_n['TOV'].sum()
#     mp_sum  = last_n['MP'].sum() or 1e-6
#     usage_rate = (fga_sum + 0.44 * fta_sum + tov_sum) / mp_sum
    
#     last_month = int(last_n['Date'].iloc[-1].month)
#     month_sin  = np.sin(2 * np.pi * last_month / 12)
#     month_cos  = np.cos(2 * np.pi * last_month / 12)
    
#     last_game_date = raw_player_df['Date'].max()
#     today = pd.Timestamp.today().normalize()
#     is_back2back = 1 if (today - last_game_date).days == 1 else 0

#     # Build the one-row DataFrame, using whatever features you like
#     vals = {
#         "Home":           is_home_game,
#         "Pace":           opp["Pace"],
#         "eFG%_y":         opp["eFG%"],
#         "TOV%":           opp["TOV%"],
#         "DRB%":           opp["DRB%"],
#         "PTS_last_5_avg": pts_avg,
#         "MP_last_5_avg":  mp_avg,
#         "PTS_trend_5":    trend,
#         "PTS_vol_5":      volatility,
#         "PTS_per_min":    eff_ewm,
#         "def_adj":        def_adj,
#         "Days_of_rest":   days_rest,
#         "usage_rate": usage_rate,
#         "month_sin" : month_sin,
#         "month_cos" : month_cos,
#         "is_back2back": is_back2back,
#     }
    
#     # filter to only the features being utilized
#     row = {col: vals[col] for col in FEATURE_COLUMNS}
#     # 5) Return as a 1×N DataFrame in the exact order the model expects
#     return pd.DataFrame([row])[FEATURE_COLUMNS]

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

    is_home_game = 1 if home == "Home" else 0

    # --- Source last N games (excluding "next" game)
    last_n = raw_player_df.tail(10).copy()
    
    # --- Rolling means
    def rolling_stat(col, w): 
        return last_n[col].shift(1).rolling(window=w, min_periods=1).mean().iloc[-1]

    def rolling_std(col): 
        return last_n[col].shift(1).rolling(window=5, min_periods=2).std().iloc[-1]

    stats = {}
    for col in ['PTS', 'FGA', '3PA', 'FTA', 'TOV', 'MP']:
        for w in [3, 5, 10]:
            stats[f'{col}_r{w}'] = rolling_stat(col, w)

    stats['PTS_std_r5'] = rolling_std('PTS')
    stats['FGA_std_r5'] = rolling_std('FGA')

    # Usage rate (manual rolling mean)
    for w in [3, 5, 10]:
        usage_vals = (
            last_n['FGA'].shift(1).rolling(w, min_periods=1).sum() +
            0.44 * last_n['FTA'].shift(1).rolling(w, min_periods=1).sum() +
            last_n['TOV'].shift(1).rolling(w, min_periods=1).sum()
        )
        mp_vals = last_n['MP'].shift(1).rolling(w, min_periods=1).sum().replace(0, 1e-6)
        stats[f'usage_rate_r{w}'] = (usage_vals / mp_vals).iloc[-1]

    # --- Efficiency & Z-score
    pts_series = last_n['PTS'].shift(1).expanding().mean()
    std_series = last_n['PTS'].shift(1).expanding().std().replace(0, 1e-6)
    stats['PTS_z'] = ((last_n['PTS'].iloc[-1] - pts_series.iloc[-1]) / std_series.iloc[-1]) if len(pts_series) > 1 else 0

    raw_eff = (last_n['PTS'] / last_n['MP'].replace(0, 1e-6)).shift(1)
    stats['PTS_per_min'] = raw_eff.ewm(span=5, adjust=False).mean().iloc[-1]

    # --- Trend (based on shifted PTS_r5 series)
    pts_r5_series = last_n['PTS'].shift(1).rolling(window=5, min_periods=1).mean()
    stats['PTS_trend_5'] = pts_r5_series.rolling(5, min_periods=2).apply(safe_polyfit, raw=False).iloc[-1]

    # --- Hot Streak
    expanding_mean = last_n['PTS'].shift(1).expanding().mean()
    hot_flag = (last_n['PTS'].shift(1) > expanding_mean).astype(int).rolling(3, min_periods=1).sum()
    stats['Hot_Streak'] = hot_flag.iloc[-1]

    # --- Calendar & Context Features
    last_game_date = raw_player_df['Date'].max()
    last_month = int(last_game_date.month)
    stats['month_sin'] = np.sin(2 * np.pi * last_month / 12)
    stats['month_cos'] = np.cos(2 * np.pi * last_month / 12)

    played = raw_player_df.loc[raw_player_df['MP'] > 0, 'Date']
    stats['Days_of_rest'] = (last_game_date - played.iloc[-2]).days if len(played) >= 2 else 0
    stats['is_back2back'] = 1 if (pd.Timestamp.today().normalize() - last_game_date).days == 1 else 0

    # --- Opponent Stats
    opp = team_stats_df.loc[team_stats_df['Team'] == opponent].iloc[0]
    stats['Opp_DRtg_x_PTSr5'] = opp['DRtg'] * stats.get('PTS_r5', 0)
    stats['Opp_Pace_x_FGAr5'] = opp['Pace'] * stats.get('FGA_r5', 0)
    stats['Opp_eFG_x_PTSr5']  = opp['eFG%'] * stats.get('PTS_r5', 0)

    league_avg = team_stats_df['DRtg'].mean()
    stats['def_adj'] = stats.get('PTS_r5', 0) * (league_avg / opp['DRtg'])

    # --- Return row aligned with model feature input
    row = {
        'Days_of_rest': stats['Days_of_rest'],
        'is_back2back': stats['is_back2back'],
        'month_sin': stats['month_sin'],
        'month_cos': stats['month_cos'],
        'PTS_r3': stats['PTS_r3'],
        'PTS_r5': stats['PTS_r5'],
        'PTS_r10': stats['PTS_r10'],
        'FGA_r3': stats['FGA_r3'],
        'FGA_r5': stats['FGA_r5'],
        'FGA_r10': stats['FGA_r10'],
        '3PA_r3': stats['3PA_r3'],
        '3PA_r5': stats['3PA_r5'],
        '3PA_r10': stats['3PA_r10'],
        'FTA_r3': stats['FTA_r3'],
        'FTA_r5': stats['FTA_r5'],
        'FTA_r10': stats['FTA_r10'],
        'TOV_r3': stats['TOV_r3'],
        'TOV_r5': stats['TOV_r5'],
        'TOV_r10': stats['TOV_r10'],
        'MP_r3': stats['MP_r3'],
        'MP_r5': stats['MP_r5'],
        'MP_r10': stats['MP_r10'],
        'usage_rate_r3': stats['usage_rate_r3'],
        'usage_rate_r5': stats['usage_rate_r5'],
        'usage_rate_r10': stats['usage_rate_r10'],
        'PTS_std_r5': stats['PTS_std_r5'],
        'FGA_std_r5': stats['FGA_std_r5'],
        'PTS_trend_5': stats['PTS_trend_5'],
        'PTS_per_min': stats['PTS_per_min'],
        'Hot_Streak': stats['Hot_Streak'],
        'PTS_z': stats['PTS_z'],
        'Opp_DRtg_x_PTSr5': stats['Opp_DRtg_x_PTSr5'],
        'Opp_Pace_x_FGAr5': stats['Opp_Pace_x_FGAr5'],
        'Opp_eFG_x_PTSr5': stats['Opp_eFG_x_PTSr5'],
        'def_adj': stats['def_adj'],
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]

print()