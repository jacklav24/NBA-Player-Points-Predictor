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

import ast
import json
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna as op
import joblib
import player_data_setup as setup
import feature_engineering as fe
from constants import COLUMNS_TO_SCALE


CSV_COLUMNS = [
    "run_id", "timestamp",
    "features", "scaled_features", "metrics", "feature_importance", "save", "player_name",
    ]
CSV_PATH = Path(__file__).parent / "data" / "optimization" / "model_runs.csv"

def run_studies(X, scaler, n_trials: int = 50):
    
    ''' Runs studies on the hyperparameters of xgb and rfr and returns the study results '''
    X_train, X_test, y_train, y_test = fe.get_train_test_splits(X)
    
    study_rfr, study_xgb, study_lgb = optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials)
    best_rfr = study_rfr.best_params
    best_xgb = study_xgb.best_params
    best_lgb = study_lgb.best_params
    return study_rfr, study_xgb, study_lgb, best_rfr, best_xgb, best_lgb



def optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials: int = 50, ):
    scaler = StandardScaler()
    
    Xs_train = setup.scale_columns(scaler, X_train.copy(), fitting=True)
    Xs_test  = setup.scale_columns(scaler, X_test.copy(),  fitting=False)
    
    # RFR objective
    def obj_rfr(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt","log2", None]),
            "random_state": 42, "n_jobs": -1
        }
        rfr_model = RandomForestRegressor(**params)
        rfr_model.fit(Xs_train, y_train)
        preds = rfr_model.predict(Xs_test)
        return root_mean_squared_error(y_test, preds)

    # XGB objective
    def obj_xgb(trial):
        params = {
            "objective":"reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42, "n_jobs": -1
        }
        xgb_model = XGBRegressor(**params)
        xgb_model.fit(Xs_train, y_train, eval_set=[(Xs_test,y_test)],
            verbose=False)
        preds = xgb_model.predict(Xs_test)
        return root_mean_squared_error(y_test, preds)


    def obj_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1
        }
        lgb_model = LGBMRegressor(**params)
        lgb_model.fit(Xs_train, y_train)
        preds = lgb_model.predict(Xs_test)
        return root_mean_squared_error(y_test, preds)
    
    ''' Here we can choose from 3 different sampler styles. Default, RandomSampler, and an explicit controlled sampler. 
        2 of the 3 should be commented out at all times.'''
    
    # 1) DEFAULT SAMPLER
    # study_rfr = op.create_study(direction="minimize", study_name="RF_RMSE")

    
    # study_rfr = op.create_study(direction="minimize", sampler=op.samplers.RandomSampler(), study_name="RF_RMSE")

    # 3) EXPLICIT CONTROL SAMPLER (requires sampler initialization line)
    sampler = op.samplers.TPESampler(n_startup_trials=10, seed=42)
    study_rfr = op.create_study(direction="minimize", sampler=sampler)
    
    # optimize!
    study_rfr.optimize(obj_rfr, n_trials=50)
    best_rfr = study_rfr.best_params


    ''' Similarly we can choose 3 sample types here. 2 of the 3 should be commented out at all times. '''
    
    # 1) DEFAULT SAMPLER
    # study_xgb = op.create_study(direction="minimize", study_name="XGB_RMSE")

    # 2) RANDOM SAMPLER
    # study_xgb = op.create_study(direction="minimize", sampler=op.samplers.RandomSampler(), study_name="XGB_RMSE") 

    # 3) EXPLICIT CONTROL SAMPLER (requires sampler initialization line)
    sampler = op.samplers.TPESampler(n_startup_trials=10, seed=42)
    study_xgb = op.create_study(direction="minimize", sampler=sampler)
    
    # optimize!
    study_xgb.optimize(obj_xgb, n_trials=50)
    best_xgb = study_xgb.best_params
    
    ''' Similarly we can choose 3 sample types here. 2 of the 3 should be commented out at all times. '''
    
    # 1) DEFAULT SAMPLER
    # study_lgb = op.create_study(direction="minimize", study_name="LGB_RMSE")

    # 2) RANDOM SAMPLER
    # study_lgb = op.create_study(direction="minimize", sampler=op.samplers.RandomSampler(), study_name="LGB_RMSE") 

    # 3) EXPLICIT CONTROL SAMPLER (requires sampler initialization line)
    sampler = op.samplers.TPESampler(n_startup_trials=10, seed=42)
    study_lgb = op.create_study(direction="minimize", sampler=sampler)
    
    # optimize!
    study_lgb.optimize(obj_lgb, n_trials=n_trials)
    best_lgb = study_lgb.best_params

    #
    with open("./data/optimization/rfr_params.json","w") as f:
        json.dump(best_rfr, f, indent=2)
    with open("./data/optimization/xgb_params.json","w") as f:
        json.dump(best_xgb, f, indent=2)
    with open("./data/optimization/lgb_params.json","w") as f:
        json.dump(best_lgb, f, indent=2)
        
    with open("./data/optimization/best_scores.json", "w") as f:
        json.dump({
        "rfr_rmse": study_rfr.best_value,
        "xgb_rmse": study_xgb.best_value,
        "lgb_rmse": study_lgb.best_value,
    }, f, indent=2)
    
    study_rfr.set_user_attr("timestamp", str(pd.Timestamp.now()))
    study_xgb.set_user_attr("timestamp", str(pd.Timestamp.now()))
    study_lgb.set_user_attr("timestamp", str(pd.Timestamp.now()))
    
    joblib.dump(study_xgb, "./data/optimization/optuna_study_xgb.pkl")
    joblib.dump(study_rfr, "./data/optimization/optuna_study_rfr.pkl")
    joblib.dump(study_lgb, "./data/optimization/optuna_study_lgb.pkl")

    # optional print statements
     
    # print("Best RF params:", study_rfr.best_params)
    # print("Best XGB params:", study_xgb.best_params)
    print("Best RF RMSE :", study_rfr.best_value)
    print("Best XGB RMSE :", study_xgb.best_value)
    print("Best LGB RMSE :", study_lgb.best_value)

    return study_rfr, study_xgb, study_lgb

def save_run(run):
    
    # (1) compute exactly as before...

    entry = {
      "run_id":            run["id"],
      "timestamp":         run["date"],
      "features":          json.dumps(run["features"]),
      "scaled_features":   json.dumps(run["scaled_features"]),
      "metrics":           json.dumps(run["metrics"]),
      "feature_importance": json.dumps(run["feature_importance"]),
      "save":              run["save"],
      "player_name":        run["player_name"],
    }

    # ensure CSV exists with header
    df_entry = pd.DataFrame([entry], columns=CSV_COLUMNS)

    # if file doesn’t exist yet, write headers; otherwise append without headers
    write_header = not CSV_PATH.exists()
    df_entry.to_csv(CSV_PATH, mode="a", index=False, header=write_header)
    
    
def get_runs():

    df = pd.read_csv(CSV_PATH)
    df['save'] = df['save'].fillna(False).astype(bool)
    df = df[df['save']]
    # turn it into a list of row‐dicts
    
    return df.to_dict(orient="records")

def get_run(runId):
    df = pd.read_csv(CSV_PATH)
    matched = df[df["run_id"] == runId]
    if matched.empty:
        return None
    run = matched.to_dict(orient="records")[0]

    for field in ("features","scaled_features","metrics","feature_importance"):
        raw = run.get(field)
        if isinstance(raw, str):
            try:
                run[field] = json.loads(raw)
            except json.JSONDecodeError:
                run[field] = ast.literal_eval(raw)

    return run