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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import player_data_setup as setup
from feature_engineering import prep_game, engineer_features
from constants import TARGET_COLUMN, FEATURE_COLUMNS, CUTOFF_VALUE


import model_metrics as mm



def load_players_data():
    base_path = "./data/all_player_rs_playoff_game_data/"
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
            if len(raw_df) < CUTOFF_VALUE:
                continue
            # 4) Preprocess & merge: parses dates, filters DNPs, merges opponent D-stats
            pre_df = setup.preprocess_player_df(raw_df, team_stats_df, player_name)

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


def train_model(player_df, rfr_params=None, xgb_params=None, lgb_params=None):
    
    y = player_df[TARGET_COLUMN]
    X = player_df[FEATURE_COLUMNS]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = setup.scale_columns(scaler, X_train, fitting=True)
    
    # Random Forest Config selection
    rfr_defaults = {"n_estimators": 100, "random_state": 42}
    rfr_cfg = rfr_defaults if rfr_params is None else {**rfr_defaults, **rfr_params}
    rfr_model = RandomForestRegressor(**rfr_cfg)

    # XGBoost
    xgb_defaults = {
        "objective":"reg:squarederror",
        "n_estimators":300, "learning_rate":0.05,
        "max_depth":8, "subsample":0.8, "colsample_bytree":0.8,
        "seed":42
    }
    xgb_cfg = xgb_defaults if xgb_params is None else {**xgb_defaults, **xgb_params}
    xgb_model = xgb.XGBRegressor(**xgb_cfg)
    
    lgb_defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    lgb_cfg = lgb_defaults if lgb_params is None else {**lgb_defaults, **lgb_params}
    lgb_model = LGBMRegressor(**lgb_cfg)
    estimators = [
    ('rf', rfr_model),
    ('xgb', xgb_model)
    ]
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(), 
        cv=5 
    )

    # Fit the models
    rfr_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)
    lgb_model.fit(X_train_scaled, y_train)
    stacked_model.fit(X_train_scaled, y_train)
    # Return the models and the splits
    return rfr_model, xgb_model, lgb_model, stacked_model, scaler, X, X_train, X_test, y_train, y_test



def get_player_list(team, df):
    filtered_df = df[df["Tm"] == team]
    return sorted(filtered_df["Player"].unique())

def get_team_list(df):
    return sorted(df["Tm"].unique())


def get_opponent_list(team_df):
    return sorted(team_df["Team"].unique())


def predict_points(player_name, team_name, opponent_name, home, full_df, team_df, model, scaler, X):
    player_games = full_df[(full_df["Player"] == player_name) & (full_df["Tm"] == team_name)]
    if player_games.empty:
        raise ValueError("No data found for selected player/team")
    
    input_row = prep_game(opponent_name, player_games, team_df, home)
 
    input_scaled = setup.scale_columns(scaler, input_row.copy(), fitting=False)

    prediction = setup.predict_game(input_scaled, scaler, model, X)
    
    return {
        "player": player_name,
        "opponent": opponent_name,
        "predicted_points": float(round(prediction, 2)),
    }
