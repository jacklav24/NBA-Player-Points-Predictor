from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from optimize import optimize_hyperparams
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json
from pathlib import Path

from xgboost import XGBRegressor

from model_logic import (
    load_players_data,
    get_player_list,
    get_team_list,
    get_opponent_list,
    train_model,
    predict_points,
)
import player_data_setup as setup

def run_studies(X_train, y_train, X_test, y_test, n_trials: int = 50):
    study_rfr, study_xgb = optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials)
    best_rfr = study_rfr.best_params
    best_xgb = study_xgb.best_params
    return study_rfr, study_xgb, best_rfr, best_xgb


def run_global_metrics(rfr_model, xgb_model, stacked_model, X_test, y_test):


    rfr_gy_pred     = rfr_model.predict(X_test)
    xgb_gy_pred     = xgb_model.predict(X_test)
    stacked_gy_pred = stacked_model.predict(X_test)

    return {
        "rfr": {
            "mae": mean_absolute_error(y_test, rfr_gy_pred),
            "rmse": root_mean_squared_error(y_test, rfr_gy_pred),
            "r2": r2_score(y_test, rfr_gy_pred),
            "bias": float((rfr_gy_pred - y_test).mean()),
        },
        "xgb": {
            "mae": mean_absolute_error(y_test, xgb_gy_pred),
            "rmse": root_mean_squared_error(y_test, xgb_gy_pred),
        },
        "stacked": {
            "mae": mean_absolute_error(y_test, stacked_gy_pred),
            "rmse": root_mean_squared_error(y_test, stacked_gy_pred),
        },
        "residuals": (y_test - stacked_gy_pred).tolist(),
        "actuals": y_test.tolist(),
        "predictions": stacked_gy_pred.tolist()
    }
    

