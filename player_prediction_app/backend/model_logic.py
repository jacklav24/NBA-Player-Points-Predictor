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
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import player_data_setup as setup
from feature_engineering import prep_game, preprocess_player_df, engineer_features

def load_players_data():
    base_path = "./data/player_game_data/"
    all_players = []
    all_teams = []
    team_stats_df = pd.read_csv("./data/team_def_stats.csv")

    # 2) Iterate over each team folder and player CSV
    for team_dir in os.listdir(base_path):
        all_teams.append(team_dir)
        team_path = os.path.join(base_path, team_dir)
        if not os.path.isdir(team_path):
            continue

        for file in os.listdir(team_path):
            if not file.endswith(".csv"):
                continue

            player_name = file.replace(".csv", "")

            # 3) Read the raw per‐game CSV into `raw_df`
            raw_df = pd.read_csv(os.path.join(team_path, file))

            # 4) Preprocess & merge: parses dates, filters DNPs, merges opponent D-stats
            pre_df = preprocess_player_df(raw_df, team_stats_df, player_name)

            # 5) Engineer all rolling & contextual features
            feat_df = engineer_features(pre_df)

            # 6) Tag player and team, then collect
            feat_df["Player"] = player_name
            feat_df["Tm"]     = team_dir
            all_players.append(feat_df)

    # 7) Concatenate every player’s feature‐engineered DataFrame
    full_df = pd.concat(all_players, ignore_index=True)

    # Return your master DataFrame plus the raw team_stats_df and list of teams
    return full_df, team_stats_df, all_teams


def train_model(player_df, rfr_params=None, xgb_params=None):
    columns_to_scale = [  "Pace",
        'PTS_last_5_avg',  'MP_last_5_avg',
            # 'PTS_5_game_trend',
            'PTS_vol_5',
            # 'Hot_Streak',
            # 'PTS_rolling_trend',
            'PTS_per_min',
            # 'PTS_pct_of_max', 
            'def_adj',
            'PTS_trend_5',
            #'Days_of_rest'
            
    ]
    to_drop = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%_x', "2P", "2PA", "2P%"
       ]
    to_test = ['Home', 'Pace', 'eFG%_y', 'TOV%', 'DRB%', 
       'PTS_last_5_avg',  'MP_last_5_avg',   'PTS_trend_5', 'PTS_vol_5', 
         "PTS_per_min", 'def_adj', ]#'Days_of_rest']
    
    exiled = ["MP_x_FGA",'FGA_last_5_avg', 'Opp_DRtg_x_PTS',
       'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS','DRtg', 'FT/FGA', "PTS_pct_of_max", "PTS_rolling_trend",'Hot_Streak',] # non-useful (for now) terms

    y = player_df["PTS"]
    X = player_df[to_test]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    X_train_scaled = setup.scale_columns(scaler, X_train, columns_to_scale, fitting=True)
    
    # Random Forest
    rf_defaults = {"n_estimators": 100, "random_state": 42}
    rf_cfg = rf_defaults if rfr_params is None else {**rf_defaults, **rfr_params}
    rfr_model = RandomForestRegressor(**rf_cfg)

    # XGBoost
    xgb_defaults = {
        "objective":"reg:squarederror",
        "n_estimators":300, "learning_rate":0.05,
        "max_depth":8, "subsample":0.8, "colsample_bytree":0.8,
        "seed":42
    }
    xgb_cfg = xgb_defaults if xgb_params is None else {**xgb_defaults, **xgb_params}
    xgb_model = xgb.XGBRegressor(**xgb_cfg)

    # rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # xgb_model = xgb.XGBRegressor(n_estimators= 313, learning_rate= 0.07064632041210707, max_depth= 3, subsample= 0.7897658742692755, colsample_bytree= 0.9969216113694006, gamma=2.200890088955729, reg_alpha= 1.4094709754246937, reg_lambda= 4.989939506224109)
  
    # objective='reg:squarederror',
    # n_estimators=300,           
    # learning_rate=0.05,           
    # max_depth=8,                
    # subsample=0.8,                
    # colsample_bytree=0.8,          
    # seed=42                       
    # )
   
    estimators = [
    ('rf', rfr_model),
    ('xgb', xgb_model)
    ]
    stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(),  # Better than plain LinearRegression
    cv=5  # Ensure proper cross-validation
    )

    rfr_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)
    stacked_model.fit(X_train_scaled, y_train)
    

    return rfr_model, xgb_model, stacked_model, scaler, columns_to_scale, X, X_train, X_test, y_train, y_test


def get_player_list(team, df):
    filtered_df = df[df["Tm"] == team]
    return sorted(filtered_df["Player"].unique())



def get_team_list(df):
    return sorted(df["Tm"].unique())


def get_opponent_list(team_df):
    return sorted(team_df["Team"].unique())


def predict_points(player_name, team_name, opponent_name, home, full_df, team_df, model, scaler, columns_to_scale, X):
    player_games = full_df[(full_df["Player"] == player_name) & (full_df["Tm"] == team_name)]
    if player_games.empty:
        raise ValueError("No data found for selected player/team")
    
    input_row = prep_game(opponent_name, player_games, team_df, player_name, home)
 
    input_scaled = setup.scale_columns(scaler, input_row.copy(), columns_to_scale, fitting=False)

    prediction = setup.predict_game(input_scaled, scaler, columns_to_scale, model, X)
    
    return {
        "player": player_name,
        "opponent": opponent_name,
        "predicted_points": float(round(prediction, 2)),
    }
