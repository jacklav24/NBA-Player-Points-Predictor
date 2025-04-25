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

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json
from pathlib import Path
import model_metrics as mm


from model_logic import (
    load_players_data,
    get_player_list,
    get_team_list,
    get_opponent_list,
    train_model,
    predict_points,
)
import player_data_setup as setup
# from feature_engineering import engineer_features, get_train_test_splits

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In‐memory cache for individual models
_indiv_model_cache = {}  # key: (player,team) → (rfr_model, xgb_model, stacked_model, scaler, X)
player = None
team = None
global rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i

rfr_model_i       = None
xgb_model_i       = None
stacked_model_i   = None
scaler_i          = None
X_i               = None
X_train_i         = None
X_test_i          = None
y_train_i         = None
y_test_i          = None

config_dir   = Path(__file__).parent / "data/optimization"
with open(config_dir / "rfr_params.json") as f:
    rfr_params = json.load(f)
with open(config_dir / "xgb_params.json") as f:
    xgb_params = json.load(f)
    
# Load and train on startup
print("[INFO] Loading data and training global model...")
players_data, team_stats, all_teams = load_players_data()

# train global models
rfr_model, xgb_model, stacked_model, scaler, X, X_train, X_test, y_train, y_test = train_model(players_data, rfr_params, xgb_params)
Xs_test = X_test.copy()
Xs_test = setup.scale_columns(scaler, X_test, False)

rfr_model_i = None
xgb_model_i = None
stacked_model_i = None
# predictions on global test set
rfr_gy_pred     = rfr_model.predict(Xs_test)
xgb_gy_pred     = xgb_model.predict(Xs_test)
stacked_gy_pred = stacked_model.predict(Xs_test)

# 1) Compute global metrics
rfr_g_mae, rfr_g_rmse = mean_absolute_error(y_test, rfr_gy_pred), root_mean_squared_error(y_test, rfr_gy_pred)
xgb_g_mae, xgb_g_rmse = mean_absolute_error(y_test, xgb_gy_pred), root_mean_squared_error(y_test, xgb_gy_pred)
stk_g_mae, stk_g_rmse = mean_absolute_error(y_test, stacked_gy_pred), root_mean_squared_error(y_test, stacked_gy_pred)

# 2) Additional stats
rfr_g_r2   = r2_score(y_test, rfr_gy_pred)
rfr_g_bias = float((rfr_gy_pred - y_test).mean())

# 3) Feature importances
rfr_g_imp = { feat: float(imp) for feat, imp in zip(X.columns, rfr_model.feature_importances_) }
xgb_g_imp = { feat: float(imp) for feat, imp in zip(X.columns, xgb_model.feature_importances_) }

# 4) Residuals / actuals / predictions (stacked model)
residuals   = [ float(err) for err in (y_test - stacked_gy_pred) ]
actuals     = [ float(a)   for a   in y_test           ]
predictions = [ float(p)   for p   in stacked_gy_pred ]

print("[INFO] Global model ready.")

def run_optimization(n_trials: int = 50):
    global X_train, y_train, X_test, y_test
    _, _, best_rfr, best_xgb = mm.run_studies(X_train, y_train, X_test, y_test)

    global rfr_model, xgb_model, stacked_model, scaler, X
    rfr_model, xgb_model, stacked_model, scaler, X, X_train, X_test, y_train, y_test  = train_model(
        players_data,
        rfr_params=best_rfr,
        xgb_params=best_xgb)
    recompute_global_metrics()
    
    print(f"🔧 Hyperparams updated: RF={best_rfr}, XGB={best_xgb}")
    return best_rfr, best_xgb

def recompute_global_metrics():
    global rfr_gy_pred, xgb_gy_pred, stacked_gy_pred, Xs_test
    global rfr_g_mae, rfr_g_rmse, xgb_g_mae, xgb_g_rmse, stk_g_mae, stk_g_rmse
    rfr_gy_pred = rfr_model.predict(Xs_test)
    xgb_gy_pred = xgb_model.predict(Xs_test)
    stacked_gy_pred = stacked_model.predict(Xs_test)


    rfr_g_mae = mean_absolute_error(y_test, rfr_gy_pred)
    rfr_g_rmse = root_mean_squared_error(y_test, rfr_gy_pred)
    xgb_g_mae = mean_absolute_error(y_test, xgb_gy_pred)
    xgb_g_rmse = root_mean_squared_error(y_test, xgb_gy_pred)
    stk_g_mae = mean_absolute_error(y_test, stacked_gy_pred)
    stk_g_rmse = root_mean_squared_error(y_test, stacked_gy_pred)
    


@app.post("/optimize")
def optimize_endpoint(n_trials: int = 30, bg: BackgroundTasks = None):
    """
    Kick off a background hyperparameter search.
    """
    try:
        bg.add_task(run_optimization, n_trials)
        return {"detail": f"Started optimization for {n_trials} trials"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_sync")
def optimize_sync(n_trials: int = 30):
    # this will block until the tuning finishes
    best_rf, best_xgb = run_optimization(n_trials)  
    return {"rf": best_rf, "xgb": best_xgb}





class PredictionRequest(BaseModel):
    player_name: str
    team: str
    opponent: str
    home: str

# API endpoints


# NBA Specific    
@app.get("/{team}/players")
def get_players(team: str):
    return get_player_list(team, players_data)

@app.get("/teams")
def get_teams():
    return get_team_list(players_data)

@app.get("/opponents")
def get_opponents():
    return get_opponent_list(team_stats)


# Prediction Endpoints

@app.post("/predict")
def run_individual_prediction(payload: PredictionRequest):
    global player, team, rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i
    player, team = payload.player_name, payload.team
    key = (player, team)
    
    try:
        indiv_df = players_data[
            (players_data["Player"] == payload.player_name) &
            (players_data["Tm"]     == payload.team)
        ].copy()
        if indiv_df.empty:
            raise HTTPException(status_code=404, detail="No data for selected player/team.")
        
        if key not in _indiv_model_cache:
            rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i = train_model(
                indiv_df, rfr_params=rfr_params, xgb_params=xgb_params
            )
        _indiv_model_cache[key] = (rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i)
        # train individual models
        # rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i,_,X_test_i,_,y_test_i = train_model(indiv_df, rfr_params, xgb_params)
        # predictions on global test set

        rfr_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, rfr_model_i, scaler_i, X_i)
        xgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, xgb_model_i, scaler_i, X_i)
        stacked_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, stacked_model_i, scaler_i, X_i)
        
        return { "rfr": rfr_pred_i, "xgb": xgb_pred_i, "stacked": stacked_pred_i }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/global_predict")
def run_global_prediction(payload: PredictionRequest):
    global player, team
    player, team = payload.player_name, payload.team
    
    
    
    try:
        # make global predictions
        rfr_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, rfr_model, scaler, X)
        xgb_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, xgb_model, scaler, X)
        stacked_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, stacked_model, scaler, X)

        return { "rfr": rfr_pred_g, "xgb": xgb_pred_g, "stacked": stacked_pred_g }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_both")
def run_blended_prediction(payload: PredictionRequest):
    global player, team, rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i
    player, team = payload.player_name, payload.team
    key = (player, team)
    
    try:
        indiv_df = players_data[
            (players_data["Player"] == payload.player_name) &
            (players_data["Tm"]     == payload.team)
        ].copy()
        if indiv_df.empty:
            raise HTTPException(status_code=404, detail="No data for selected player/team.")
        
        if key not in _indiv_model_cache:
            rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i = train_model(
                indiv_df, rfr_params=rfr_params, xgb_params=xgb_params
            )
        _indiv_model_cache[key] = (rfr_model_i, xgb_model_i, stacked_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i)
        
        # make individual predictions
        rfr_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, rfr_model_i, scaler_i, X_i)
        xgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, xgb_model_i, scaler_i, X_i)
        stacked_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, stacked_model_i, scaler_i, X_i)
        # make global predictions
        rfr_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, rfr_model, scaler, X)
        xgb_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, xgb_model, scaler, X)
        stacked_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, stacked_model, scaler, X)


        # --- 3) Compute the blend weight α based on N games ---
        N = len(indiv_df)
        THRESHOLD = 30           # tweak this to taste
        α = min(1.0, N/THRESHOLD)

        # --- 4) Blend the three predictions ---
        blended = {
          "rfr":     α*rfr_pred_i["predicted_points"]    + (1-α)*rfr_pred_g["predicted_points"],
          "xgb":     α*xgb_pred_i["predicted_points"]    + (1-α)*xgb_pred_g["predicted_points"],
          "stacked": α*stacked_pred_i["predicted_points"]+ (1-α)*stacked_pred_g["predicted_points"],
        }
        to_return = {
          "global_model":     {"rfr": rfr_pred_g,     "xgb": xgb_pred_g,     "stacked": stacked_pred_g},
          "individual_model": {"rfr": rfr_pred_i,      "xgb": xgb_pred_i,      "stacked": stacked_pred_i},
          "blended_model":    {"rfr": round(blended["rfr"],2), 
                                "xgb": round(blended["xgb"],2), 
                                "stacked": round(blended["stacked"],2),
                                "alpha": round(α, 2)}
        }
        print(to_return)
        # --- 5) Return all three sets side by side ---
        return {
          "global_model":     {"rfr": rfr_pred_g,     "xgb": xgb_pred_g,     "stacked": stacked_pred_g},
          "individual_model": {"rfr": rfr_pred_i,      "xgb": xgb_pred_i,      "stacked": stacked_pred_i},
          "blended_model":    {"rfr": round(blended["rfr"],2), 
                                "xgb": round(blended["xgb"],2), 
                                "stacked": round(blended["stacked"],2),
                                "alpha": round(α, 2)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Model Info Endpoints

# @app.get("/model_insights")
# def model_insights():#payload: PredictionRequest):
#     global Xs_test, player, team, scaler_i
#     rfr_p_g = rfr_model.predict(Xs_test)
#     xgb_p_g = xgb_model.predict(Xs_test)
#     stk_p_g = stacked_model.predict(Xs_test)
#     key = (player, team)
#     Xs_test_i = None
    
#     if key not in _indiv_model_cache:
#         return out
#     (
#       rfr_model_i,
#       xgb_model_i,
#       stacked_model_i,
#       scaler_i,
#       X_i,
#       X_train_i,
#       X_test_i,
#       y_train_i,
#       y_test_i
#     ) = _indiv_model_cache[key]
    
    

#     out = {
#         "metrics": {
#             "rfr_mae_g": mean_absolute_error(y_test, rfr_p_g),
#             "rfr_rmse_g": root_mean_squared_error(y_test, rfr_p_g),
#             "rfr_r2_g": r2_score(y_test, rfr_p_g),
#             "rfr_bias_g": float((rfr_p_g - y_test).mean()),

#             "xgb_mae_g": mean_absolute_error(y_test, xgb_p_g),
#             "xgb_rmse_g": root_mean_squared_error(y_test, xgb_p_g),
#             "xgb_r2_g": r2_score(y_test, xgb_p_g),
#             "xgb_bias_g": float((xgb_p_g - y_test).mean()),

#             "stacked_mae_g": mean_absolute_error(y_test, stk_p_g),
#             "stacked_rmse_g": root_mean_squared_error(y_test, stk_p_g),
#             "stacked_r2_g": r2_score(y_test, stk_p_g),
#             "stacked_bias_g": float((stk_p_g - y_test).mean()),
#         },
#         "feature_importance": {
#             "rfr": rfr_g_imp,
#             "xgb": xgb_g_imp
#         }
#     }
    

#     if rfr_model_i is not None:
#        # scale only the configured columns
#         Xs_test_i = X_test_i.copy()
#         Xs_test_i = setup.scale_columns(scaler_i, X_test_i, False)

#         rfr_p_i = rfr_model_i.predict(Xs_test_i)
#         out["metrics"].update({
#             "rfr_mae_i": mean_absolute_error(y_test_i, rfr_p_i),
#             "rfr_rmse_i": root_mean_squared_error(y_test_i, rfr_p_i),
#             "rfr_r2_i": r2_score(y_test_i, rfr_p_i),
#             "rfr_bias_i": float((rfr_p_i - y_test_i).mean()),
#         })

#     if xgb_model_i is not None:
#         Xs_test_i = X_test_i.copy()
#         Xs_test_i = setup.scale_columns(scaler_i, X_test_i, False)
#         xgb_p_i = xgb_model_i.predict(Xs_test_i)
#         out["metrics"].update({
#             "xgb_mae_i": mean_absolute_error(y_test_i, xgb_p_i),
#             "xgb_rmse_i": root_mean_squared_error(y_test_i, xgb_p_i),
#             "xgb_r2_i": r2_score(y_test_i, xgb_p_i),
#             "xgb_bias_i": float((xgb_p_i - y_test_i).mean()),
#         })

#     if stacked_model_i is not None:
#         Xs_test_i = X_test_i.copy()
#         Xs_test_i = setup.scale_columns(scaler_i, X_test_i, False)
#         stk_p_i = stacked_model_i.predict(Xs_test_i)
#         out["metrics"].update({
#             "stacked_mae_i": mean_absolute_error(y_test_i, stk_p_i),
#             "stacked_rmse_i": root_mean_squared_error(y_test_i, stk_p_i),
#             "stacked_r2_i": r2_score(y_test_i, stk_p_i),
#             "stacked_bias_i": float((stk_p_i - y_test_i).mean()),
#         })
    
#     return out


@app.get("/model_insights")
def model_insights():
    """
    Returns metrics and feature importances for the global models,
    and—if an individual model has been trained for the last queried player/team—
    metrics for the individual and blended models as well.
    """
    # --- 1) Compute global metrics on the original full dataset's hold-out set ---
    # Scale the global test set (from initial train_test_split on full_df)
    
    # Predictions from global models on that global hold-out
    global Xs_test, player, team, scaler_i
    rfr_p_g = rfr_model.predict(Xs_test)
    xgb_p_g = xgb_model.predict(Xs_test)
    stk_p_g = stacked_model.predict(Xs_test)

    # Build base output with global metrics
    out = {
        "metrics": {
            "rfr_mae_g":  mean_absolute_error(y_test, rfr_p_g),
            "rfr_rmse_g": root_mean_squared_error(y_test, rfr_p_g),
            "rfr_r2_g":   r2_score(y_test, rfr_p_g),
            "rfr_bias_g": float((rfr_p_g - y_test).mean()),

            "xgb_mae_g":  mean_absolute_error(y_test, xgb_p_g),
            "xgb_rmse_g": root_mean_squared_error(y_test, xgb_p_g),
            "xgb_r2_g":   r2_score(y_test, xgb_p_g),
            "xgb_bias_g": float((xgb_p_g - y_test).mean()),

            "stacked_mae_g":  mean_absolute_error(y_test, stk_p_g),
            "stacked_rmse_g": root_mean_squared_error(y_test, stk_p_g),
            "stacked_r2_g":   r2_score(y_test, stk_p_g),
            "stacked_bias_g": float((stk_p_g - y_test).mean()),
        },
        "feature_importance": {
            "rfr": rfr_g_imp,
            "xgb": xgb_g_imp
        }
    }

    # --- 2) Check if we have an individual model cached ---
    key = (player, team)
    if key not in _indiv_model_cache:
        return out

    # --- 3) Unpack individual model and its test split ---
    (
        rfr_i, xgb_i, stk_i,
        scaler_i, X_i,
        X_train_i, X_test_i,
        y_train_i, y_test_i
    ) = _indiv_model_cache[key]

    # --- 4) Compute individual predictions and metrics on that player's hold-out ---
    Xs_indiv = setup.scale_columns(scaler_i, X_test_i.copy(), fitting=False)
    rfr_p_i = rfr_i.predict(Xs_indiv)
    xgb_p_i = xgb_i.predict(Xs_indiv)
    stk_p_i = stk_i.predict(Xs_indiv)
    out["metrics"].update({
        "rfr_mae_i":  mean_absolute_error(y_test_i, rfr_p_i),
        "rfr_rmse_i": root_mean_squared_error(y_test_i, rfr_p_i),
        "rfr_r2_i":   r2_score(y_test_i, rfr_p_i),
        "rfr_bias_i": float((rfr_p_i - y_test_i).mean()),

        "xgb_mae_i":  mean_absolute_error(y_test_i, xgb_p_i),
        "xgb_rmse_i": root_mean_squared_error(y_test_i, xgb_p_i),
        "xgb_r2_i":   r2_score(y_test_i, xgb_p_i),
        "xgb_bias_i": float((xgb_p_i - y_test_i).mean()),

        "stacked_mae_i":  mean_absolute_error(y_test_i, stk_p_i),
        "stacked_rmse_i": root_mean_squared_error(y_test_i, stk_p_i),
        "stacked_r2_i":   r2_score(y_test_i, stk_p_i),
        "stacked_bias_i": float((stk_p_i - y_test_i).mean()),
    })

    # --- 5) Compute blended metrics: use the individual's hold-out, but global preds from that same set ---
    alpha = min(1.0, len(X_i) / 30)
    global_on_indiv = {
        'rfr':  rfr_model.predict(Xs_indiv),
        'xgb':  xgb_model.predict(Xs_indiv),
        'stacked': stacked_model.predict(Xs_indiv)
    }
    indiv_preds = {'rfr': rfr_p_i, 'xgb': xgb_p_i, 'stacked': stk_p_i}

    for m in ['rfr', 'xgb', 'stacked']:
        blended = alpha * indiv_preds[m] + (1 - alpha) * global_on_indiv[m]
        out["metrics"].update({
            f"{m}_mae_b":  mean_absolute_error(y_test_i, blended),
            f"{m}_rmse_b": root_mean_squared_error(y_test_i, blended),
            f"{m}_r2_b":   r2_score(y_test_i, blended),
            f"{m}_bias_b": float((blended - y_test_i).mean()),
        })

    return out


#  FUTURE POTENTIAL ENDPOINTS

# @app.get("/tuning_results")
# def tuning_results():
#     try:
#         study_rfr = joblib.load("optuna_study_rfr.pkl")
#         study_xgb = joblib.load("optuna_study_xgb.pkl")

#         def trials_to_dict(study):
#             return [{
#                 "trial": t.number,
#                 "value": t.value,
#                 "params": t.params
#             } for t in study.trials if t.state.name == "COMPLETE"]

#         return {
#             "rfr": {
#                 "best_params": study_rfr.best_params,
#                 "best_value": study_rfr.best_value,
#                 "trials": trials_to_dict(study_rfr),
#             },
#             "xgb": {
#                 "best_params": study_xgb.best_params,
#                 "best_value": study_xgb.best_value,
#                 "trials": trials_to_dict(study_xgb),
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/reset_individual_models")
# def reset_individual_models():
    # global rfr_model_i, xgb_model_i, stacked_model_i
    # rfr_model_i = None
    # xgb_model_i = None
    # stacked_model_i = None
    # return {"status": "cleared"}