import json
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import optuna as op
import joblib



def run_studies(X_train, y_train, X_test, y_test, n_trials: int = 50):
    ''' Runs studies on the hyperparameters of xgb and rfr and returns the study results '''
    study_rfr, study_xgb = optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials)
    best_rfr = study_rfr.best_params
    best_xgb = study_xgb.best_params
    return study_rfr, study_xgb, best_rfr, best_xgb



def optimize_hyperparams(X_train, y_train, X_test, y_test, n_trials: int = 50, ):

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
        rfr_model.fit(X_train, y_train)
        preds = rfr_model.predict(X_test)
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
        xgb_model.fit(X_train, y_train, eval_set=[(X_test,y_test)],
            verbose=False)
        preds = xgb_model.predict(X_test)
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

    #
    with open("./data/optimization/rfr_params.json","w") as f:
        json.dump(best_rfr, f, indent=2)
    with open("./data/optimization/xgb_params.json","w") as f:
        json.dump(best_xgb, f, indent=2)
        
    with open("./data/optimization/best_scores.json", "w") as f:
        json.dump({
        "rfr_rmse": study_rfr.best_value,
        "xgb_rmse": study_xgb.best_value
    }, f, indent=2)
    
    study_rfr.set_user_attr("timestamp", str(pd.Timestamp.now()))
    study_xgb.set_user_attr("timestamp", str(pd.Timestamp.now()))
    
    joblib.dump(study_xgb, "optuna_study_xgb.pkl")
    joblib.dump(study_rfr, "optuna_study_rfr.pkl")

    # optional print statements
    
    # print("Best RF params:", study_rfr.best_params)
    # print("Best XGB params:", study_xgb.best_params)
    # print("Best RF RMSE :", study_rfr.best_value)
    # print("Best XGB RMSE :", study_xgb.best_value)

    return study_rfr, study_xgb
    
    
