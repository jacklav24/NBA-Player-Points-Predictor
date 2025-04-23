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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from model_logic import (
    load_players_data,
    get_player_list,
    get_team_list,
    get_opponent_list,
    train_model,
    predict_points,
)
import player_data_setup as setup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and train on startup
print("[INFO] Loading data and training global model...")
players_data, team_stats, all_teams = load_players_data()
# train global models
rfr_model, xgb_model, stacked_model, scaler, columns_to_scale, X, X_train, X_test, y_train, y_test = train_model(players_data)

# predictions on global test set
rfr_gy_pred     = rfr_model.predict(X_test)
xgb_gy_pred     = xgb_model.predict(X_test)
stacked_gy_pred = stacked_model.predict(X_test)

# 1) Compute global metrics
rfr_g_mae, rfr_g_rmse = mean_absolute_error(y_test, rfr_gy_pred), root_mean_squared_error(y_test, rfr_gy_pred)
xgb_g_mae, xgb_g_rmse = mean_absolute_error(y_test, xgb_gy_pred), root_mean_squared_error(y_test, xgb_gy_pred)
stk_g_mae, stk_g_rmse = mean_absolute_error(y_test, stacked_gy_pred), root_mean_squared_error(y_test, stacked_gy_pred)

# 2) Additional stats
rfr_g_r2   = r2_score(y_test, rfr_gy_pred)
rfr_g_bias = float((rfr_gy_pred - y_test).mean())

# 3) Feature importances
rfr_imp = { feat: float(imp) for feat, imp in zip(X.columns, rfr_model.feature_importances_) }
xgb_imp = { feat: float(imp) for feat, imp in zip(X.columns, xgb_model.feature_importances_) }

# 4) Residuals / actuals / predictions (stacked model)
residuals   = [ float(err) for err in (y_test - stacked_gy_pred) ]
actuals     = [ float(a)   for a   in y_test           ]
predictions = [ float(p)   for p   in stacked_gy_pred ]

print("[INFO] Global model ready.")

class PredictionRequest(BaseModel):
    player_name: str
    team: str
    opponent: str
    home: str


def get_model_stats(model, scaler, columns_to_scale, X_test, y_test):
    """
    Retrains each model's own hold-out set to compute MAE/RMSE.
    """
    Xs = X_test.copy()
    Xs[columns_to_scale] = scaler.transform(Xs[columns_to_scale])
    y_pred = model.predict(Xs)
    return mean_absolute_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred)

# API endpoints
@app.get("/{team}/players")
def get_players(team: str):
    return get_player_list(team, players_data)

@app.get("/teams")
def get_teams():
    return get_team_list(players_data)

@app.get("/opponents")
def get_opponents():
    return get_opponent_list(team_stats)

@app.post("/predict")
def run_individual_prediction(payload: PredictionRequest):
    try:
        indiv_df = players_data[
            (players_data["Player"] == payload.player_name) &
            (players_data["Tm"]     == payload.team)
        ]
        if indiv_df.empty:
            raise HTTPException(status_code=404, detail="No data for selected player/team.")

        # train individual models
        rfr_i, xgb_i, stk_i, scaler_i, cols_i, X_i, Xtr_i, Xte_i, ytr_i, yte_i = train_model(indiv_df)

        # make predictions
        res_rfr = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, rfr_i, scaler_i, cols_i, X_i)
        res_xgb = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, xgb_i, scaler_i, cols_i, X_i)
        res_stk = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, stk_i, scaler_i, cols_i, X_i)

        # attach metrics computed on individual hold-out
        res_rfr["mae"], res_rfr["rmse"] = get_model_stats(rfr_i, scaler_i, cols_i, Xte_i, yte_i)
        res_xgb["mae"], res_xgb["rmse"] = get_model_stats(xgb_i, scaler_i, cols_i, Xte_i, yte_i)
        res_stk["mae"], res_stk["rmse"] = get_model_stats(stk_i, scaler_i, cols_i, Xte_i, yte_i)

        return { "rfr": res_rfr, "xgb": res_xgb, "stacked": res_stk }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/global_predict")
def run_global_prediction(payload: PredictionRequest):
    try:
        # make global predictions
        g_rfr = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, rfr_model, scaler, columns_to_scale, X)
        g_xgb = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, xgb_model, scaler, columns_to_scale, X)
        g_stk = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, stacked_model, scaler, columns_to_scale, X)

        # attach global metrics
        g_rfr["mae"], g_rfr["rmse"] = rfr_g_mae, rfr_g_rmse
        g_xgb["mae"], g_xgb["rmse"] = xgb_g_mae, xgb_g_rmse
        g_stk["mae"], g_stk["rmse"] = stk_g_mae,   stk_g_rmse

        return { "rfr": g_rfr, "xgb": g_xgb, "stacked": g_stk }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_insights")
def model_insights():
    return {
        "metrics": {
            "rfr_mae":   round(rfr_g_mae,  2),
            "xgb_mae":   round(xgb_g_mae,  2),
            "stacked_mae": round(stk_g_mae, 2),
            "r2":        round(r2_score(y_test, stacked_gy_pred), 4),
            "bias":      round(float((stacked_gy_pred - y_test).mean()), 2)
        },
        "feature_importance": { "rfr": rfr_imp, "xgb": xgb_imp },
        "residuals":   residuals,
        "actuals":     actuals,
        "predictions": predictions
    }

