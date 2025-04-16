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
from model_logic import (
    load_players_data,
    get_player_list,
    get_team_list,
    get_opponent_list,
    train_model,
    predict_points,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and train on startup
print("[INFO] Loading data and training global model...")
players_data, team_stats, all_teams = load_players_data()
rfr_model, xgb_model, stacked_model, scaler, columns_to_scale, X = train_model(players_data)
print("[INFO] Global model ready.")


class PredictionRequest(BaseModel):
    player_name: str
    team: str
    opponent: str

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
        individual_data = players_data[
            (players_data["Player"] == payload.player_name) &
            (players_data["Tm"] == payload.team)
        ]
        if individual_data.empty:
            raise HTTPException(status_code=404, detail="No individual data found for the selected player/team.")
        
        rfr_indiv, xgb_indiv, stacked_indiv, scaler_indiv, columns_indiv, X_indiv = train_model(individual_data)

        indiv_rfr = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            individual_data,
            team_stats,
            rfr_indiv,
            scaler_indiv,
            columns_indiv,
            X_indiv
        )
        indiv_xgb = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            individual_data,
            team_stats,
            xgb_indiv,
            scaler_indiv,
            columns_indiv,
            X_indiv
        )
        indiv_stacked = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            individual_data,
            team_stats,
            stacked_indiv,
            scaler_indiv,
            columns_indiv,
            X_indiv
        )

        return {
            "rfr": indiv_rfr,
            "xgb": indiv_xgb,
            "stacked": indiv_stacked
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/global_predict")
def run_global_prediction(payload: PredictionRequest):
    try:
        global_rfr = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            players_data,
            team_stats,
            rfr_model,
            scaler,
            columns_to_scale,
            X
        )
        global_xgb = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            players_data,
            team_stats,
            xgb_model,
            scaler,
            columns_to_scale,
            X
        )
        global_stacked = predict_points(
            payload.player_name,
            payload.team,
            payload.opponent,
            players_data,
            team_stats,
            stacked_model,
            scaler,
            columns_to_scale,
            X
        )

        return {
            "rfr": global_rfr,
            "xgb": global_xgb,
            "stacked": global_stacked
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
