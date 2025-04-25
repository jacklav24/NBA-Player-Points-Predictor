import json
import joblib
import optuna as op
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from feature_engineering import preprocess_player_df, engineer_features, get_train_test_splits
from model_logic import load_players_data, train_model  # adjust imports as needed
from xgboost import XGBRegressor
# 



def optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials: int = 50, ):
    # 1) Prepare data
    
    # 2) RF objective
    def obj_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt","log2", None]),
            "random_state": 42, "n_jobs": -1
        }
        m = RandomForestRegressor(**params)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        return root_mean_squared_error(y_test, preds)

    # 3) XGB objective (similar)
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
        m = XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_test,y_test)],
            verbose=False)
        preds = m.predict(X_test)
        return root_mean_squared_error(y_test, preds)

    # 4) Run studies
    #Initial bayesian optimization
    # study_rfr = op.create_study(direction="minimize", study_name="RF_RMSE")

    #random search optimization
    # study_rfr = op.create_study(direction="minimize", sampler=op.samplers.RandomSampler(), study_name="RF_RMSE")

    #explicit control optimization
    sampler = op.samplers.TPESampler(n_startup_trials=10, seed=42)
    study_rfr = op.create_study(direction="minimize", sampler=sampler)
    study_rfr.optimize(obj_rf, n_trials=50)
    best_rfr = study_rfr.best_params

    # study_xgb = op.create_study(direction="minimize", study_name="XGB_RMSE")

    # THIS is the Random Search version
    # study_xgb = op.create_study(direction="minimize", sampler=op.samplers.RandomSampler(), study_name="XGB_RMSE")

    #This is the explicit sampler tuning.
    sampler = op.samplers.TPESampler(n_startup_trials=10, seed=42)
    study_xgb = op.create_study(direction="minimize", sampler=sampler)
    
    study_xgb.optimize(obj_xgb, n_trials=50)
    best_xgb = study_xgb.best_params

    # 5) Persist to disk
    with open("data/optimization/rfr_params.json","w") as f:
        json.dump(best_rfr, f, indent=2)
    with open("data/optimization/xgb_params.json","w") as f:
        json.dump(best_xgb, f, indent=2)
        
    with open("data/optimization/best_scores.json", "w") as f:
        json.dump({
        "rfr_rmse": study_rfr.best_value,
        "xgb_rmse": study_xgb.best_value
    }, f, indent=2)
    study_rfr.set_user_attr("timestamp", str(pd.Timestamp.now()))
    study_xgb.set_user_attr("timestamp", str(pd.Timestamp.now()))
    
    joblib.dump(study_xgb, "optuna_study_xgb.pkl")
    joblib.dump(study_rfr, "optuna_study_rfr.pkl")

    # print("✍️ Best RF params:", study_rfr.best_params)
    # print("✍️ Best XGB params:", study_xgb.best_params)
    # print("🔢 Best RF RMSE :", study_rfr.best_value)
    # print("🔢 Best XGB RMSE :", study_xgb.best_value)

    return study_rfr, study_xgb
    
    
