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
    merged_df["MP"] = merged_df["MP"].apply(convert_mp)
    merged_df['PTS'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['MP'] = pd.to_numeric(merged_df['MP'], errors='coerce')
    merged_df['FGA'] = pd.to_numeric(merged_df['PTS'], errors='coerce')
    merged_df['PTS'] = merged_df["PTS"].astype(float)
    merged_df['MP'] = merged_df["MP"].astype(float)
    merged_df['FGA'] = merged_df["FGA"].astype(float)
    return merged_df[col_to_keep]

def convert_mp(mp):
    minutes, seconds = mp.split(':')
    return int(minutes) + int(seconds) / 60

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
    
    player_df['PTS_per_minute'] = player_df['PTS'] / player_df['MP']
    player_df['PTS_pct_of_max'] = player_df['PTS_last_5_avg'] / player_df['PTS'].expanding().max()
    
    player_df = player_df.drop(columns=['PTS_5_MA'], errors='ignore')
    
    return player_df
def get_splits(player_df):
    to_drop = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%_x', "2P", "2PA", "2P%"
       ]
    to_test = ['Home', 'Pace', 'eFG%_y', 'TOV%', 'DRB%', 
       'PTS_last_5_avg',  'MP_last_5_avg',   'PTS_5_game_trend', 'PTS_volatility_5', 
       'Hot_Streak',  "PTS_per_minute", "PTS_pct_of_max", "PTS_rolling_trend"]
    
    exiled = ["MP_x_FGA",'FGA_last_5_avg', 'Opp_DRtg_x_PTS',
       'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS','DRtg', 'FT/FGA',] # non-useful (for now) terms

    y = player_df['PTS']  # Target variable

    X = player_df[to_test]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X, X_test, X_train, y_train, y_test

def prep_game(opponent, player, teams):
    """Prepares features for prediction while maintaining column name compatibility"""
    # Get opponent stats
    opp_stats = teams[teams['Team'] == opponent].iloc[0]
    
    # Use last 10 games (all historical)
    last_10 = player.iloc[-10:].copy() if len(player) >= 10 else player.copy()
    
    # Clean numeric columns
    for col in ['PTS', 'FGA', 'MP']:
        last_10[col] = pd.to_numeric(last_10[col], errors='coerce').fillna(0)
        last_10[col] = last_10[col].astype(float if col == 'MP' else int)
    
    # Calculate rolling averages (matches get_rolling_avgs)
    pts_avg = last_10['PTS'].rolling(window=10, min_periods=1).mean().iloc[-1]
    fga_avg = last_10['FGA'].rolling(window=10, min_periods=1).mean().iloc[-1]
    mp_avg = last_10['MP'].rolling(window=10, min_periods=1).mean().iloc[-1]
    
    # Improved trend calculation (but stored in PTS_5_game_trend column)
    last_5_ma = last_10['PTS'].rolling(5).mean()
    pts_trend = last_5_ma.diff().mean()  # Average change between MA periods
    
    # Volatility (matches original)
    pts_volatility = last_10['PTS'].rolling(window=10, min_periods=1).std().shift(1).fillna(0).iloc[-1]

    # Hot streak calculation (matches original)
    hot_streak = (
        (last_10['PTS'] > last_10['PTS'].expanding().mean().shift(1))
        .astype(int)
        .rolling(window=3)
        .sum()
        .iloc[-2]  # equivalent to shift(1)
    )
    
    # Efficiency metrics (matches original column names)
    last_game = last_10.iloc[-1]
    pts_per_minute = last_game['PTS'] / last_game['MP'] if last_game['MP'] > 0 else 0
    pts_pct_of_max = pts_avg / player['PTS'].max() if player['PTS'].max() > 0 else 0
    
    # Build feature row with original column names
    return pd.DataFrame({
        'Home': 0,  # Assuming away game
        # 'DRtg': opp_stats['DRtg'],
        'Pace': opp_stats['Pace'],
        'eFG%_y': opp_stats['eFG%'],
        'TOV%': opp_stats['TOV%'],
        'DRB%': opp_stats['DRB%'],
        # 'FT/FGA': opp_stats['FT/FGA'],
        'PTS_last_5_avg': pts_avg,
        # 'FGA_last_5_avg': fga_avg,
        'MP_last_5_avg': mp_avg,
        # 'Opp_DRtg_x_PTS': opp_stats['DRtg'] * pts_avg,
        # 'Opp_Pace_x_FGA': opp_stats['Pace'] * fga_avg,
        # 'Opp_eFG_x_PTS': opp_stats['eFG%'] * pts_avg,
        'PTS_5_game_trend': pts_trend,
        'PTS_volatility_5': pts_volatility,
        'Hot_Streak': hot_streak,
        'PTS_per_minute': pts_per_minute,
        'PTS_pct_of_max': pts_pct_of_max,
        'PTS_rolling_trend': pts_trend  # Duplicate to match both names
    }, index=[0])

def predict_game(input_row, scaler, columns_to_scale, model, X):

    input_row_scaled = input_row.copy()
    input_row_scaled = input_row_scaled[X.columns]
    input_row_scaled[columns_to_scale] = scaler.transform(input_row[columns_to_scale])
    
    predicted_pts = model.predict(input_row)[0]
    return predicted_pts
   

    