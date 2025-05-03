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

from datetime import datetime
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
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
from constants import FEATURE_COLUMNS, COLUMNS_TO_SCALE

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load static data
players_data, team_stats, all_teams = load_players_data()

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
def recompute_global_metrics():
    global rfr_gy_pred, xgb_gy_pred, lgb_gy_pred, stk_gy_pred, Xs_test
    global rfr_g_mae, rfr_g_rmse, xgb_g_mae, xgb_g_rmse, lgb_g_mae, lgb_g_rmse, stk_g_mae, stk_g_rmse

    rfr_gy_pred = rfr_model.predict(Xs_test)
    xgb_gy_pred = xgb_model.predict(Xs_test)
    lgb_gy_pred = lgb_model.predict(Xs_test)
    stk_gy_pred = stacked_model.predict(Xs_test)

    rfr_g_mae = mean_absolute_error(y_test, rfr_gy_pred)
    rfr_g_rmse = root_mean_squared_error(y_test, rfr_gy_pred)
    xgb_g_mae = mean_absolute_error(y_test, xgb_gy_pred)
    xgb_g_rmse = root_mean_squared_error(y_test, xgb_gy_pred)
    lgb_g_mae  = mean_absolute_error(y_test, lgb_gy_pred)
    lgb_g_rmse = root_mean_squared_error(y_test, lgb_gy_pred)
    stk_g_mae  = mean_absolute_error(y_test, stk_gy_pred)
    stk_g_rmse = root_mean_squared_error(y_test, stk_gy_pred)


# Load global models on startup
def load_global_models():
    try:
        rfr      = joblib.load(MODEL_DIR/"rfr_global.pkl")
        xgb      = joblib.load(MODEL_DIR/"xgb_global.pkl")
        lgb      = joblib.load(MODEL_DIR/"lgb_global.pkl")
        stk  = joblib.load(MODEL_DIR/"stacked_global.pkl")
        scaler   = joblib.load(MODEL_DIR/"scaler_global.pkl")
        X        = joblib.load(MODEL_DIR/"X_global.pkl")
        X_test   = joblib.load(MODEL_DIR/"X_test_global.pkl")
        y_test   = joblib.load(MODEL_DIR/"y_test_global.pkl")
        return rfr, xgb, lgb, stk, scaler, X, X_test, y_test
    except FileNotFoundError:
        return None, None, None, None, None, None, None


rfr_model, xgb_model, lgb_model, stacked_model, scaler, X, X_test, y_test = load_global_models()
if rfr_model is not None:
    # scale exactly once
    Xs_test = setup.scale_columns(scaler, X_test.copy(), fitting=False)
    recompute_global_metrics()
else:
    print("[INFO] No saved global models found - will train when needed")
    rfr_model = xgb_model = lgb_model = stacked_model = scaler = X = None
    X_train = X_test = y_train = y_test = Xs_test = None
# if rfr_model is None:
#     train_global()
    

# In-memory cache for individual models
player = None
team = None
_indiv_model_cache = {}

# Load hyperparameter configs
config_dir = Path(__file__).parent / "data" / "optimization"
with open(config_dir / "rfr_params.json") as f:
    rfr_params = json.load(f)
with open(config_dir / "xgb_params.json") as f:
    xgb_params = json.load(f)
with open(config_dir / "lgb_params.json") as f:
    lgb_params = json.load(f)



# Train global model on demand
@app.post("/train_global")
def train_global(n_trials: int = 50):
    global rfr_model, xgb_model, lgb_model, stacked_model, scaler, X
    global X_train, X_test, y_train, y_test, Xs_test
    # 1) Train from scratch
    (
        rfr_model, xgb_model, lgb_model, stacked_model,
        scaler, X,
        X_train, X_test, y_train, y_test
    ) = train_model(players_data, rfr_params, xgb_params, lgb_params)

    # 2) Scale test set
    Xs_test = setup.scale_columns(scaler, X_test.copy(), fitting=False)

    # 3) Compute and cache global metrics
    recompute_global_metrics()

    # 4) Persist models and data for next startup
    joblib.dump(rfr_model, MODEL_DIR / "rfr_global.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgb_global.pkl")
    joblib.dump(lgb_model, MODEL_DIR / "lgb_global.pkl")
    joblib.dump(stacked_model, MODEL_DIR / "lgb_global.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_global.pkl")
    joblib.dump(X, MODEL_DIR / "X_global.pkl")
    joblib.dump(X_test,      MODEL_DIR / "X_test_global.pkl")
    joblib.dump(y_test,      MODEL_DIR / "y_test_global.pkl")
    joblib.dump(X_train,      MODEL_DIR / "X_train_global.pkl")
    joblib.dump(y_train,      MODEL_DIR / "y_train_global.pkl")
    joblib.dump(Xs_test, MODEL_DIR / "Xs_test_global.pkl")

    return {"detail": "Global model retrained"}


def compute_diagnostics_for_test_set(
    Xs_test, y_test,
    models,
    individual_split=None,
    alpha=None,
    skip_global: bool = False,
):
    """
    models = {
      'global':    (rfr_model, xgb_model, lgb_model),
      'individual': (rfr_model_i, xgb_model_i, lgb_model_i)
    }
    individual_split = (X_i_test, y_i_test)
    alpha = blend weight (0–1) for blended metrics
    """
    out = {'metrics':{}, 'feature_importance':{}}
    if not skip_global:
        # --- GLOBAL metrics & importances (suffix _g) ---
        for tag, mdl in zip(['rfr','xgb','lgb', 'stk'], models['global']):
            preds = mdl.predict(Xs_test)
            out['metrics'].update({
            f'{tag}_mae_g':  mean_absolute_error(y_test, preds),
            f'{tag}_rmse_g': root_mean_squared_error(y_test, preds),
            f'{tag}_r2_g':   r2_score(y_test, preds),
            f'{tag}_bias_g': float((preds - y_test).mean()),
            f'{tag}_within_n_g': within_n_points(y_test, preds, n=3),
            })
            if tag in ('rfr','xgb', 'lgb'):
                out['feature_importance'][f'{tag}_g'] = {
                feat: float(imp)
                for feat, imp in zip(Xs_test.columns, mdl.feature_importances_)
                }


    # Bail if no individual model
    if 'individual' not in models or individual_split is None:
        return out

    rfr_i, xgb_i, lgb_i, stk_i = models['individual']
    X_i_test, y_i_test   = individual_split

    # --- INDIVIDUAL metrics & importances (suffix _i) ---
    for tag, mdl in zip(['rfr','xgb','lgb', 'stk'], [rfr_i, xgb_i, lgb_i, stk_i]):
        preds = mdl.predict(X_i_test)
        out['metrics'].update({
          f'{tag}_mae_i':  mean_absolute_error(y_i_test, preds),
          f'{tag}_rmse_i': root_mean_squared_error(y_i_test, preds),
          f'{tag}_r2_i':   r2_score(y_i_test, preds),
          f'{tag}_bias_i': float((preds - y_i_test).mean()),
          f'{tag}_within_n_i': within_n_points(y_i_test, preds, n=3),
        })
        if tag in ('rfr','xgb', 'lgb'):
            out['feature_importance'][f'{tag}_i'] = {
              feat: float(imp)
              for feat, imp in zip(X_i_test.columns, mdl.feature_importances_)
            }

    # --- BLENDED metrics (suffix _b) ---
    # requires an alpha blend weight
    if alpha is not None:
        for tag, (ind_mdl, glob_mdl) in zip(
            ['rfr','xgb','lgb', 'stk'],
            [(rfr_i, models['global'][0]),
             (xgb_i, models['global'][1]),
             (lgb_i, models['global'][2]),
             (stk_i, models['global'][3]),]
        ):
            indiv_preds = ind_mdl.predict(X_i_test)
            glob_preds  = glob_mdl.predict(X_i_test)
            blended     = alpha*indiv_preds + (1-alpha)*glob_preds

            out['metrics'].update({
              f'{tag}_mae_b':      mean_absolute_error(y_i_test, blended),
              f'{tag}_rmse_b':     root_mean_squared_error(y_i_test, blended),
              f'{tag}_r2_b':       r2_score(y_i_test, blended),
              f'{tag}_bias_b':     float((blended - y_i_test).mean()),
              f'{tag}_within_n_b': within_n_points(y_i_test, blended, n=3),
            })

    return out


def run_optimization(n_trials: int = 50):
    global X_train, y_train, Xs_test, y_test
    global rfr_model, xgb_model, lgb_model, stacked_model, scaler, players_data
    _, _, _, best_rfr, best_xgb, best_lgb = mm.run_studies(players_data, scaler)

    
    rfr_model, xgb_model, lgb_model, stacked_model, scaler, X, X_train, X_test, y_train, y_test  = train_model(
        players_data,
        rfr_params=best_rfr,
        xgb_params=best_xgb,
        lgb_params=best_lgb)
    Xs_test = setup.scale_columns(scaler, X_test.copy(), False)
    recompute_global_metrics()
    
    print(f"🔧 Hyperparams updated: RF={best_rfr}, XGB={best_xgb}")
    return best_rfr, best_xgb

    
def within_n_points(y_true, y_pred, n=5):
    differences = np.abs(y_true - y_pred)
    return np.mean(differences <= n)  # returns percentage




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
    save_run: bool

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
    global player, team, rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i
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
            rfr_model_i, xgb_model_i, lgb_model_i,  stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i = train_model(
                indiv_df, rfr_params=rfr_params, xgb_params=xgb_params, lgb_params=lgb_params
            )
        _indiv_model_cache[key] = (rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i)

        rfr_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, rfr_model_i, scaler_i, X_i)
        xgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, xgb_model_i, scaler_i, X_i)
        lgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, lgb_model_i, scaler_i, X_i)
        stk_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, stk_model_i, scaler_i, X_i)
        
        return { "rfr": rfr_pred_i, "xgb": xgb_pred_i, "lgb": lgb_pred_i, "stk": stk_pred_i }

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
        lgb_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, lgb_model, scaler, X)
        stk_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, stacked_model, scaler, X)

        return { "rfr": rfr_pred_g, "xgb": xgb_pred_g, "lgb": lgb_pred_g, "stk" : stk_pred_g }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_both")
def run_all_prediction(payload: PredictionRequest):
    global player, team, rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i
    global Xs_test, player, team, scaler_i
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
            rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i = train_model(
                indiv_df, rfr_params=rfr_params, xgb_params=xgb_params, lgb_params=lgb_params
            )
        _indiv_model_cache[key] = (rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i)
        
        # make individual predictions
        rfr_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, rfr_model_i, scaler_i, X_i)
        xgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, xgb_model_i, scaler_i, X_i)
        lgb_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, lgb_model_i, scaler_i, X_i)
        stk_pred_i = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                  indiv_df, team_stats, stk_model_i, scaler_i, X_i)
        # make global predictions
        rfr_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, rfr_model, scaler, X)
        xgb_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, xgb_model, scaler, X)
        lgb_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, lgb_model, scaler, X)
        stk_pred_g = predict_points(payload.player_name, payload.team, payload.opponent, payload.home,
                                 players_data, team_stats, stacked_model, scaler, X)

        # --- 3) Compute the blend weight α based on N games ---
        N = len(indiv_df)
        THRESHOLD = 30           # tweak this to taste
        α = min(1.0, N/THRESHOLD)
        α = (N/30) / (1 + N/30)

        # --- 4) Blend the three predictions ---
        blended = {
          "rfr":     α*rfr_pred_i["predicted_points"]    + (1-α)*rfr_pred_g["predicted_points"],
          "xgb":     α*xgb_pred_i["predicted_points"]    + (1-α)*xgb_pred_g["predicted_points"],
          "lgb": α*lgb_pred_i["predicted_points"]+ (1-α)*lgb_pred_g["predicted_points"],
          "stk": α*stk_pred_i["predicted_points"]+ (1-α)*stk_pred_g["predicted_points"],
        }
        Xs_test_i = X_test_i

        diagnostics = compute_diagnostics_for_test_set(
                Xs_test, y_test,
                models={
                    'global':    (rfr_model, xgb_model, lgb_model, stacked_model),
                    'individual':(rfr_model_i, xgb_model_i, lgb_model_i, stk_model_i)
                },
                individual_split=(Xs_test_i, y_test_i),
                alpha=α
                )


        # 4) build your run record
        run = {
        "id":       str(uuid4()),
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": FEATURE_COLUMNS,
        "scaled_features": COLUMNS_TO_SCALE,
        "metrics": diagnostics["metrics"],
        "feature_importance": diagnostics["feature_importance"],
        "save": payload.save_run,
        "player_name" : payload.player_name
        }

        if payload.save_run:
            mm.save_run(run)

        # 5) return predictions as JSON
        return {
        "global_model":     {"rfr": rfr_pred_g, "xgb": xgb_pred_g, "lgb": lgb_pred_g, "stk": stk_pred_g},
        "individual_model": {"rfr": rfr_pred_i, "xgb": xgb_pred_i, "lgb": lgb_pred_i, "stk": stk_pred_i},
        "blended_model":    {"rfr": round(blended["rfr"],2), 
                                "xgb": round(blended["xgb"],2), 
                                "lgb": round(blended["lgb"],2),
                                "stk": round(blended["stk"],2),
                                "alpha": round(α, 2)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Model Info Endpoints


@app.get("/model_insights")
def model_insights():
    # Declare globals
    global rfr_model, xgb_model, lgb_model, stacked_model, scaler, X, Xs_test, y_test, player, team



    if rfr_model is None or Xs_test is None:
        raise HTTPException(400, "Global model not trained. POST /train_global first.")
    diagnostics = compute_diagnostics_for_test_set(
        Xs_test, y_test,
        models={"global": (rfr_model, xgb_model, lgb_model, stacked_model)},
        individual_split=None, alpha=None
    )

    # Include individual/blended if available
    key = (player, team)
    if key in _indiv_model_cache:
        rfr_i, xgb_i, lgb_i, stk_i, scaler_i, X_i, X_train_i, X_test_i, y_train_i, y_test_i = _indiv_model_cache[key]
        Xs_i_test = setup.scale_columns(scaler_i, X_test_i.copy(), fitting=False)
        extra = compute_diagnostics_for_test_set(
            Xs_i_test, y_test_i,
            models={
                "global": (rfr_model, xgb_model, lgb_model, stacked_model),
                "individual": (rfr_i, xgb_i, lgb_i, stk_i)
            },
            individual_split=(Xs_i_test, y_test_i),
            alpha=(len(X_test_i)/30)/(1 + len(X_test_i)/30),
            skip_global=True
        )
        diagnostics["metrics"].update(extra["metrics"])
        diagnostics["feature_importance"].update(extra["feature_importance"])
    # print(diagnostics)
   
    return diagnostics

@app.get("/get_runs")
def get_runs():
    return mm.get_runs()

@app.get("/run-history/{runId}")
def get_run_details(runId: str):
    print(f"getting run details for {runId}")
    run = mm.get_run(runId)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
    
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
    # global rfr_model_i, xgb_model_i, lgb_model_i
    # rfr_model_i = None
    # xgb_model_i = None
    # lgb_model_i = None
    # return {"status": "cleared"}