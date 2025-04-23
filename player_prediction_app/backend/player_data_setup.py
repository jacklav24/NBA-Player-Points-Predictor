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
from sklearn.model_selection import train_test_split
import numpy as np
from feature_engineering import preprocess_player_df, engineer_features

FEATURE_COLUMNS = ['Home', 
    'Pace', 'eFG%_y', 'TOV%', 'DRB%',
    'PTS_last_5_avg', 'MP_last_5_avg', 'PTS_trend_5',
    'PTS_vol_5', 'PTS_per_min', 'def_adj', 'Days_of_rest'
]

def read_misc_stats(filepath):
    team_stats_df = pd.read_csv(filepath)


    team_stats_df = team_stats_df.drop(columns=['W','L','PW','PL','MOV','SOS','SRS','ORtg','FTr','3PAr','eFG%','TOV%','ORB%','SOS','SRS','ORtg','FTr', '3PAr', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA'])
    team_stats_df.rename(columns={'eFG%.1': 'eFG%', 'TOV%.1': 'TOV%', 'FT/FGA.1':'FT/FGA'}, inplace=True)
    team_stats_df.to_csv(filepath, index=False)
    return

def player_data_merge(player_file):
    
    df = pd.read_csv(player_file)
    df2 = pd.read_csv('./data/team_def_stats.csv')
    df['Home'] = df['Unnamed: 5'].apply(lambda x: 1 if pd.isna(x) or x == '' else 0)
    df = df[df['MP'].notna() & (df['MP'] != '') & (df["MP"] != 'Inactive') & (df["MP"] != 'Did Not Play') & (df["MP"] != 'Did Not Dress') & (df["MP"] != 'Not With Team')]
    merged_df = pd.merge(df, df2, left_on='Opp', right_on='Team', how='left')
    col_to_keep = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
       'eFG%_x', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK',
       'TOV', 'PF', 'PTS', 'Home',
       'DRtg', 'Pace', 'eFG%_y', 'TOV%', 'DRB%', 'FT/FGA']
    # merged_df["MP"] = merged_df["MP"].apply(convert_mp)
    merged_df['PTS'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['MP'] = pd.to_numeric(merged_df['MP'], errors='coerce')
    merged_df['FGA'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['PTS'] = merged_df["PTS"].astype(float)
    merged_df['MP'] = merged_df["MP"].astype(float)
    merged_df['FGA'] = merged_df["FGA"].astype(float)
    return merged_df[col_to_keep]


def scale_columns(scaler, X, columns, fitting=False):
    if fitting:
        X[columns] = scaler.fit_transform(X[columns])
    else:
        X[columns] = scaler.transform(X[columns])
        
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

def predict_game(input_row, scaler, columns_to_scale, model, X):
    
    input_row_scaled = input_row.copy()
    input_row_scaled = input_row_scaled[X.columns]
    input_row_scaled[columns_to_scale] = scaler.transform(input_row[columns_to_scale])
    
    predicted_pts = model.predict(input_row)[0]
    return predicted_pts
   

    