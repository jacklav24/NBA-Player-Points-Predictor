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
import pandas as pd
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

def load_players_data():
    base_path = "./data/player_game_data/"
    all_players = []
    all_teams = []

    for team_dir in os.listdir(base_path):
        all_teams.append(team_dir)
        team_path = os.path.join(base_path, team_dir)
        if not os.path.isdir(team_path):
            continue

        for file in os.listdir(team_path):
            if file.endswith(".csv"):
                player_name = file.replace(".csv", "")
                df = setup.player_data_merge(os.path.join(team_path, file))
                df["Player"] = player_name
                df["Tm"] = team_dir
                all_players.append(df)

    full_df = pd.concat(all_players, ignore_index=True)
    full_df = setup.get_rolling_avgs(full_df)
    team_stats = pd.read_csv("./data/team_def_stats.csv")
    return full_df, team_stats, all_teams


def train_model(player_df):
    columns_to_scale = [  "Pace",
        'PTS_last_5_avg',  'MP_last_5_avg',
            'PTS_5_game_trend',
            'PTS_volatility_5',
            'Hot_Streak',
            'PTS_rolling_trend',
            'PTS_per_minute',
            'PTS_pct_of_max',
    ]
    exiled = ['MP_x_FGA', 'FGA_last_5_avg', 'Opp_DRtg_x_PTS',
        'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS','DRtg', 'FT/FGA', "TOV%", "DRB%",] # stats that were unhelpful or overfitting
    

    X, X_test, X_train, y_train, y_test = setup.get_splits(player_df)
    scaler = StandardScaler()

    X_train_scaled = setup.scale_columns(scaler, X_train, columns_to_scale, fitting=True)
    rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,           
    learning_rate=0.05,           
    max_depth=8,                
    subsample=0.8,                
    colsample_bytree=0.8,          
    seed=42                       
    )
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
    
    X_test_scaled = setup.scale_columns(scaler, X_test, columns_to_scale)
    # rfr_y_pred = rfr_model.predict(X_test_scaled)
    # xgb_y_pred = xgb_model.predict(X_test_scaled)
    # stacked_y_pred = stacked_model.predict(X_test_scaled)

    # rfr_mae = mean_absolute_error(y_test, rfr_y_pred)
    # xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
    # stacked_mae = mean_absolute_error(y_test, stacked_y_pred)

    # rfr_rmse = root_mean_squared_error(y_test, rfr_y_pred)
    # xgb_rmse = root_mean_squared_error(y_test, xgb_y_pred)
    # stacked_rmse = root_mean_squared_error(y_test, stacked_y_pred)
    

    return rfr_model, xgb_model, stacked_model, scaler, columns_to_scale, X


def get_player_list(team, df):
    filtered_df = df[df["Tm"] == team]
    return sorted(filtered_df["Player"].unique())



def get_team_list(df):
    return sorted(df["Tm"].unique())


def get_opponent_list(team_df):
    return sorted(team_df["Team"].unique())


def predict_points(player_name, team_name, opponent_name, full_df, team_df, model, scaler, columns_to_scale, X):
    player_games = full_df[(full_df["Player"] == player_name) & (full_df["Tm"] == team_name)]
    if player_games.empty:
        raise ValueError("No data found for selected player/team")

    input_row = setup.prep_game(opponent_name, player_games, team_df)
    input_scaled = setup.scale_columns(scaler, input_row.copy(), columns_to_scale, fitting=False)

    prediction = setup.predict_game(input_scaled, scaler, columns_to_scale, model, X)
    
    # Evaluate model with known test data for metrics
    X, X_test, X_train, y_train, y_test = setup.get_splits(full_df)
    X_test_scaled = X_test.copy()
    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    
    return {
        "player": player_name,
        "opponent": opponent_name,
        "predicted_points": float(round(prediction, 2)),
        "mae": float(round(mae, 2)),
        "rmse": float(round(rmse, 2))
    }