import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import pandas as pd

# --- Import your existing functions ---
from player_data_setup import preprocess_player_df
from model_logic import engineer_features, load_players_data
from model_metrics import run_studies
from feature_engineering import engineer_features


def build_player_datasets(min_games_cutoff: int):
    """
    Load all raw CSVs, preprocess each player's data, engineer features,
    and include only players with at least `min_games_cutoff` games.
    """
    players_data, team_stats, all_teams = load_players_data()
    # load_players_data returns raw players_data; we will re-filter by cutoff

    frames = []
    for player in players_data['Player'].unique():
        sub = players_data[players_data['Player'] == player]
        if len(sub) < min_games_cutoff:
            continue
        # we assume engineer_features takes preprocessed df
        # print("this is how many players we're looking at", len(sub))
        df = engineer_features(sub)
        frames.append(df)

    if not frames:
        raise ValueError(f"No players with >= {min_games_cutoff} games found.")
    return pd.concat(frames, ignore_index=True)


def sweep_cutoff_hpo(output_path: str = "cutoff_sweep_results.json"):
    # define the cutoffs to test
    cutoffs = [5, 10, 15, 20, 25, 30, 40]
    results = {}

    for cutoff in cutoffs:
        print(f"\n=== Testing cutoff = {cutoff} games ===")
        # build dataset for this cutoff
        data = build_player_datasets(cutoff)
        # separate into features + target
        X = data   # adjust if your target column name differs
        y = data["PTS"]

        # run hyperparam search for this X, y
        study_rfr, study_xgb, _, _ = run_studies(X, None, n_trials=50)

        # record best params and best RMSE values
        results[cutoff] = {
            "rfr_params": study_rfr.best_params,
            "rfr_rmse":   study_rfr.best_value,
            "xgb_params": study_xgb.best_params,
            "xgb_rmse":   study_xgb.best_value
        }
        # write intermediate results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Cutoff {cutoff} done. RMSEs: RF {study_rfr.best_value:.3f}, XGB {study_xgb.best_value:.3f}")

    print("\nSweep complete. Results saved to", output_path)


if __name__ == "__main__":
    sweep_cutoff_hpo()
