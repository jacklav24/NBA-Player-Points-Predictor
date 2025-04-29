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
import re

from constants import TARGET_COLUMN, FEATURE_COLUMNS, COLUMNS_TO_SCALE

# Keywords indicating Did Not Play
DNP_KEYWORDS = {'Inactive', 'Did Not Play', 'Did Not Dress', 'Not With Team', ''}

SHOOTING_STATS = {
    'FG%': 'FGA',   # Field Goal %
    '3P%': '3PA',   # 3-Point %
    '2P%': '2PA',   # 2-Point %
    'FT%': 'FTA'    # Free Throw %
}

def read_misc_stats(filepath):
    team_stats_df = pd.read_csv(filepath)


    team_stats_df = team_stats_df.drop(columns=['W','L','PW','PL','MOV','SOS','SRS','ORtg','FTr','3PAr','eFG%','TOV%','ORB%','SOS','SRS','ORtg','FTr', '3PAr', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA'])
    team_stats_df.rename(columns={'eFG%.1': 'eFG%', 'TOV%.1': 'TOV%', 'FT/FGA.1':'FT/FGA'}, inplace=True)
    team_stats_df.to_csv(filepath, index=False)
    return

# converts minutes played from 00:00 format to minutes (with decimal)
def convert_mp(mp):
    minutes, seconds = mp.split(':')
    return int(minutes) + int(seconds) / 60


# def safe_convert_mp(x):
#     # only convert when it’s a string like “00:00” or “5:23”
#     if isinstance(x, str) and re.fullmatch(r'\d{1,2}:\d{2}', x):
#         return convert_mp(x)
#     return x
def safe_convert_mp(x):
    if isinstance(x, str):
        x = x.strip()
        # MM:SS → float minutes
        if re.fullmatch(r'\d{1,2}:\d{2}', x):
            return convert_mp(x)
        # pure integer string → minutes
        if re.fullmatch(r'\d+', x):
            return float(x)
        # empty or non-numeric → NaN
        return pd.NA
    # already numeric → leave alone
    return x
    
def preprocess_player_df(df: pd.DataFrame, team_stats_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """
    Raw CSV → merged, cleaned DataFrame ready for feature engineering.
    - Parses dates, converts MP string→float, filters out DNPs.
    - Merges opponent team defensive stats.
    """
    # print("player df columns", df.columns)
    df = df.copy()
    df["Player"] = player_name
    num_cols = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # df['MP'] = (
    # df['MP']
    #   .astype(str)
    #   .str.split(':').apply(lambda ms: int(ms[0]) + int(ms[1])/60 if ':' in ms else pd.to_numeric(ms, errors='coerce'))
    # )
    # df['MP'] = pd.to_numeric(df['MP'], errors='coerce').fillna(0)

    # merge player game logs with opposing team's defensive stats
    merged = df.merge(team_stats_df, left_on='Opp', right_on='Team', how='left')
    # print('merged_columns', merged.columns)
    # Parse date
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    
    # Flag home games
    # merged['Home'] = merged['Unnamed: 5'].apply(lambda x: 1 if pd.isna(x) or x == '' else 0)
    possible = ["Unnamed: 5", "Unnamed: 2"]
    home_col = next((c for c in possible if c in merged.columns), None)
    merged["Home"] = merged[home_col] \
        .apply(lambda x: 1 if pd.isna(x) or x == "" else 0)
    # drop games where 
    
    # Convert MP to minutes float
    # merged['MP'] = merged['MP'].apply(lambda x: safe_convert_mp(x) if isinstance(x, str) else 0)

    # Numeric coercion
    merged['PTS'] = pd.to_numeric(merged['PTS'], errors='coerce').fillna(0)
    merged['FGA'] = pd.to_numeric(merged['FGA'], errors='coerce').fillna(0)
    merged['MP'] = pd.to_numeric(merged['FGA'], errors='coerce').fillna(0)
    merged.drop(columns=['Rk', 'Gcar', 'Gtm', 'Team_x', 'Unnamed: 5', 'Unnamed: 2', 'Opp', 'Result',
        'GmSc', '+/-', 'Unnamed: 0', 'Team_y', ], inplace=True, errors='ignore')
    # merged.drop(columns=['Rk', 'Gcar', 'Gtm', 'Team_x', 'Unnamed: 5', 'Opp', 'Result',
    # 'GmSc', '+/-', 'Unnamed: 0', 'Team_y', ], inplace=True)
    # # Filter out Did Not Play
    # df = df[~df['MP'].isin([0])]
    for pct_col, attempts_col in SHOOTING_STATS.items():
        # Set percentage to 0 if attempts = 0 (no shots → 0% success)
        merged.loc[merged[attempts_col] == 0, pct_col] = 0
        
        # Force-convert to numeric (strings/empty → NaN)
        merged[pct_col] = pd.to_numeric(merged[pct_col], errors='coerce')
        
        # Fill remaining NaN (missing/invalid) with 0
        merged[pct_col] = merged[pct_col].fillna(0)

    
    return merged

def scale_columns(scaler, X, fitting=False):
    ''' FITTING = FALSE : if you are doing the INITIAL Scaling of the data. 
        FITTING = TRUE  : if you are setting up the scaler'''
        
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if fitting:
        X[COLUMNS_TO_SCALE] = scaler.fit_transform(X[COLUMNS_TO_SCALE])
    else:
        X[COLUMNS_TO_SCALE] = scaler.transform(X[COLUMNS_TO_SCALE])
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def compute_trend(series):
    if len(series) < 2:
        return 0  # or np.nan if you prefer
    return np.polyfit(np.arange(len(series)), series, 1)[0]


def get_rolling_avgs(player_df):
    # Original rolling averages
    player_df['PTS_last_5_avg'] = player_df['PTS'].rolling(window=5, min_periods=3).mean()  # Changed from 10→5
    player_df['FGA_last_5_avg'] = player_df['FGA'].rolling(window=10, min_periods=1).mean()
    player_df['MP_last_5_avg'] = player_df['MP'].rolling(window=10, min_periods=1).mean()
    
    # Interaction terms
    player_df['Opp_DRtg_x_PTS'] = player_df['DRtg'] * player_df['PTS_last_5_avg']
    player_df['Opp_Pace_x_FGA'] = player_df['Pace'] * player_df['FGA_last_5_avg']
    player_df['Opp_eFG_x_PTS'] = player_df['eFG%_y'] * player_df['PTS_last_5_avg']
    
    # Improved trend calculations
    player_df['PTS_5_MA'] = player_df['PTS'].rolling(window=5, min_periods=3).mean()
    
    # New: Rolling trend (slope of last 5 MA values)
    player_df['PTS_rolling_trend'] = (
    player_df['PTS_5_MA']
    .rolling(window=5, min_periods=2)
    .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
    .fillna(
        player_df['PTS_5_MA'].expanding(min_periods=2)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan)
    )
    .fillna(0)
    )

    # Fill any remaining NaN values (first row if min_periods=2)
    player_df['PTS_rolling_trend'] = player_df['PTS_rolling_trend'].fillna(0)  # or another appropriate value

    player_df['PTS_5_game_trend'] = player_df['PTS_5_MA'].diff().fillna(0)
    player_df['PTS_volatility_5'] = player_df['PTS'].rolling(window=10, min_periods=1).std().shift(1).fillna(0)

    player_df['Hot_Streak'] = (
        (player_df['PTS'] > player_df['PTS'].expanding().mean().shift(1))
        .astype(int)
        .rolling(window=3, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
    )
       # smoothing efficiency: raw and 5‑game rolling
    player_df['PTS_per_minute_raw'] = player_df['PTS'] / player_df['MP'].replace(0, np.nan)
    player_df['PTS_per_minute']     = player_df['PTS_per_minute_raw'].rolling(5, min_periods=1).mean().fillna(0)
    player_df['PTS_per_minute'] = (
    0.5*player_df['PTS_per_minute'] + 
    0.5*(player_df['PTS'] / player_df['MP'].replace(0,np.nan))
).fillna(0)
    player_df.drop(columns=['PTS_per_minute_raw'], inplace=True)

    # player_df['PTS_per_minute'] = player_df['PTS'] / player_df['MP']
    player_df['PTS_pct_of_max'] = player_df['PTS_last_5_avg'] / player_df['PTS'].expanding().max()
    avg_def_rtg = player_df['DRtg'].mean()

# Then, for each game, scale the player’s last‐5 pts by how “easy” or “hard” the opponent D was:
    player_df['def_adj'] = (
    player_df['PTS_last_5_avg'] * (avg_def_rtg / player_df['DRtg']))
    player_df = player_df.drop(columns=['PTS_5_MA'], errors='ignore')
    
    return player_df

def predict_game(input_row, scaler, model, X):
    
    input_row_scaled = input_row.copy()
    input_row_scaled = input_row_scaled[X.columns]
    input_row_scaled[COLUMNS_TO_SCALE] = scaler.transform(input_row[COLUMNS_TO_SCALE])
    
    predicted_pts = model.predict(input_row)[0]
    return predicted_pts
   

    