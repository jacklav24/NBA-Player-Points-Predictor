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
os.makedirs("plots", exist_ok=True)
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import player_data_setup as setup
from feature_engineering import prep_game, engineer_features
from constants import TARGET_COLUMN, FEATURE_COLUMNS, CUTOFF_VALUE


import model_metrics as mm
from pathlib import Path

BASE_DIR = Path(__file__).parent
SEASON_DATA_SOURCES = (
    ("2024", BASE_DIR / "data" / "player_game_data", BASE_DIR / "data" / "team_def_stats.csv"),
    ("2025", BASE_DIR / "data" / "player_game_data_2025", BASE_DIR / "data" / "team_def_stats_2025.csv"),
)


# def load_players_data():
#     # base_path = "./data/player_game_data/"
#     base_paths = [
#         ("./data/player_game_data/", "./data/team_def_stats.csv", "2024"),
#         ("./data/player_game_data_2025/", "./data/team_def_stats_2025.csv", "2025"),
#     ]
#     all_players = []
#     all_teams = []
#     team_stats_df = pd.read_csv("./data/team_def_stats.csv")

#     # 2) Iterate over each team folder and player CSV
#     for team_dir in os.listdir(base_path):
#         all_teams.append(team_dir)
#         team_path = os.path.join(base_path, team_dir)
#         if not os.path.isdir(team_path):
#             continue

#         for file in os.listdir(team_path):
#             if not file.endswith(".csv"):
#                 continue

#             player_name = file.replace(".csv", "")

#             # 3) Read the raw per‐game CSV into `raw_df`
#             # raw_df = pd.read_csv(os.path.join(team_path, file))
#             try:
#                 raw_df = pd.read_csv(os.path.join(team_path, file))
                
#             except pd.errors.ParserError as e:
#                 print(f"ParserError in file: {os.path.join(team_path, file)}")
#                 print(e)
#             if len(raw_df) < CUTOFF_VALUE:
#                 continue
#             # 4) Preprocess & merge: parses dates, filters DNPs, merges opponent D-stats
#             pre_df = setup.preprocess_player_df(raw_df, team_stats_df, player_name)

#             # 5) Engineer all rolling & contextual features
#             feat_df = engineer_features(pre_df)

#             # 6) Tag player and team, then collect
#             feat_df["Player"] = f"{player_name}_{season}"
#             feat_df["Tm"]     = team_dir
#             all_players.append(feat_df)

#     # 7) Concatenate every player’s feature‐engineered DataFrame
#     full_df = pd.concat(all_players, ignore_index=True)

#     # Return your master DataFrame plus the raw team_stats_df and list of teams
#     return full_df, team_stats_df, all_teams


def load_players_data(include_seasons=None):
    """
    Build the unified training dataframe. Each season listed in
    SEASON_DATA_SOURCES loads with its dedicated defensive stats file so that,
    for example, 2025 player logs are merged with `team_def_stats_2025.csv`.
    """
    requested_seasons = (
        set(include_seasons) if include_seasons else {season for season, *_ in SEASON_DATA_SOURCES}
    )

    all_players = []
    all_teams = set()
    team_stats_by_season = {}

    for season, base_path, def_path in SEASON_DATA_SOURCES:
        if season not in requested_seasons:
            continue

        if not base_path.exists():
            continue  # nothing to load for this season

        if not def_path.exists():
            raise FileNotFoundError(
                f"Expected defensive stats for {season} at {def_path}, but the file was not found."
            )

        team_stats_df = pd.read_csv(def_path)
        team_stats_by_season[season] = team_stats_df

        for team_dir in os.listdir(base_path):
            team_path = os.path.join(base_path, team_dir)
            if not os.path.isdir(team_path):
                continue

            all_teams.add(team_dir)

            for file in os.listdir(team_path):
                if not file.endswith(".csv"):
                    continue

                player_name = file.replace(".csv", "")
                file_path = os.path.join(team_path, file)

                try:
                    raw_df = pd.read_csv(file_path)
                except pd.errors.ParserError as e:
                    print(f"ParserError in file: {file_path}")
                    print(e)
                    continue

                if len(raw_df) < CUTOFF_VALUE:
                    continue

                pre_df = setup.preprocess_player_df(raw_df, team_stats_df, player_name)
                feat_df = engineer_features(pre_df)

                feat_df["Player"] = f"{player_name}_{season}"
                feat_df["Tm"] = team_dir
                feat_df["Season"] = season
                all_players.append(feat_df)

    if not all_players:
        raise ValueError("No player data found for the requested seasons.")

    if not team_stats_by_season:
        raise ValueError("No defensive team stats were loaded for the requested seasons.")

    full_df = pd.concat(all_players, ignore_index=True)

    if "2025" in team_stats_by_season:
        preferred_team_stats = team_stats_by_season["2025"]
    else:
        preferred_team_stats = next(iter(team_stats_by_season.values()))

    return full_df, preferred_team_stats, sorted(all_teams)


def train_model(player_df, rfr_params=None, xgb_params=None, lgb_params=None):
    
    y = player_df[TARGET_COLUMN]
    X = player_df[FEATURE_COLUMNS]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    
    X_train_scaled = setup.scale_columns(scaler, X_train, fitting=True)
    X_scaled = setup.scale_columns(scaler, X, fitting=False)
    X_test_scaled = setup.scale_columns(scaler, X_test, fitting=False)
    
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
        ('rfr', rfr_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
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
    
    models = {
        'rfr': rfr_model,
        'xgb': xgb_model,
        'lgb': lgb_model,
        'stk': stacked_model
    }

    # for tag, mdl in models.items():
    #     mdl.fit(X_train_scaled, y_train)

    #     # 1. Train/Test metrics
    #     train_preds = mdl.predict(X_train_scaled)
    #     test_preds  = mdl.predict(X_test_scaled)
    #     print(f"\n[{tag.upper()}] Train MAE: {mean_absolute_error(y_train, train_preds):.3f}")
    #     print(f"[{tag.upper()}] Test  MAE:  {mean_absolute_error(y_test,  test_preds):.3f}")
    #     print(f"[{tag.upper()}] Test  R²:   {r2_score(y_test, test_preds):.3f}")

    #     # 2. Cross-validation
    #     cv_scores = cross_val_score(mdl, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    #     print(f"[{tag.upper()}] CV MAE:    {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    #     # 3. Learning Curve
    #     train_sizes, train_scores, val_scores = learning_curve(
    #         mdl, X_scaled, y, cv=5, scoring='neg_mean_absolute_error', train_sizes=np.linspace(0.1, 1.0, 5)
    #     )
    #     plt.figure()
    #     plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train MAE')
    #     plt.plot(train_sizes, -val_scores.mean(axis=1), label='Val MAE')
    #     plt.title(f"{tag.upper()} Learning Curve")
    #     plt.xlabel("Training Size")
    #     plt.ylabel("MAE")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"plots/{tag}_learning_curve.png")
    #     plt.close()

    #     # 4. Residual Plot
    #     residuals = y_test - test_preds
    #     plt.figure()
    #     plt.scatter(test_preds, residuals, alpha=0.6)
    #     plt.axhline(0, color='r', linestyle='--')
    #     plt.title(f"{tag.upper()} Residuals")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Residual")
    #     plt.tight_layout()
    #     plt.savefig(f"plots/{tag}_residuals.png")
    #     plt.close()

    #     # 5. Permutation Importance
    #     perm = permutation_importance(mdl, X_test_scaled, y_test, scoring='neg_mean_absolute_error', n_repeats=5)
    #     sorted_idx = perm.importances_mean.argsort()
    #     plt.figure()
    #     plt.barh(range(len(sorted_idx)), perm.importances_mean[sorted_idx])
    #     plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    #     plt.title(f"{tag.upper()} Permutation Importance")
    #     plt.tight_layout()
    #     plt.savefig(f"plots/{tag}_perm_importance.png")
    #     plt.close()


    # Return the models and the splits
    return rfr_model, xgb_model, lgb_model, stacked_model, scaler, X, X_train, X_test, y_train, y_test



def get_player_list(team, df, season=2025):
    filtered_df = df[df["Tm"] == team]
    filtered_df = filtered_df[filtered_df["Player"].str.endswith(f"_{season}")]
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
