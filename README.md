# NBA Player Points Predictor

This is a full-stack web app that predicts NBA player points based on past game performance and opponent defense metrics. Part originally of a stats project in school, but something I thought I'd build out some UI for.

It's a great chance to improve and look into modeling, machine learning, and potentially neural networks in the future.

## Folder Structure

There are 3 main directories/folders. 

1. backend
    - This houses the python and model logic. Hosts endpoints for training models, tuning hyperparameters, and making predictions.
    - currently the backend needs to be shuffled around so that file structure makes a little more sense. functionality is perfect though.
2. frontend
    - Holds react frontend including pages, components, and endpoints to post/get model info for the user.
3. data_collection
    - Holds a jupyter notebook which utilizes the nba_api to get all active players data for model training.


```
player-predictor-app/
├── backend/
│   ├── main.py                  # FastAPI entrypoint
│   ├── model_logic.py           # Model logic and data loading
│   ├── player_data_setup.py     # Provided preprocessing functions
│   ├── model_metrics.py         # Calculates and saves model metrics.
|   ├── feature_engineering.py   # handles the chunk of feature engineering
|   ├── constants.py             # holds constants related to model training and the project as whole.
│   ├── data/
│   │   ├── team_def_stats.csv
│   │   └── player_game_logs/
│   │       └── {TEAM}/
│   │           └── {PLAYER}.csv
│   └── requirements.txt
│
├── frontend/
│   ├── index.html               # Entry HTML
│   ├── package.json             # Project dependencies
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── components/          # Custom react components
│   │   ├── pages/               # Front-end react pages
│   │   └── index.css            # Tailwind styles
│
├── data_collection/
│   ├── data_collection_adjustment.ipynb    # Jupyter Notebook to take and transform data
├── README.md
```

---

## 🚀 How to Run

### Backend
```zsh
cd player_predition_app/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install fastapi uvicorn scikit-learn pandas numpy
uvicorn main:app --reload
```

### Frontend
```zsh
cd player_prediction_app/frontend
npm install
npm run dev
```

Frontend will start on `http://localhost:5173` and communicate with FastAPI at `http://localhost:8000`.

---

## ✅ Features
### - Select your player, team, and opponent for game prediction
### - Run prediction and view predicted single game points. 
        - The global model is trained of EVERY player, EVERY game this year. The individual is trained on ONLY the selected player. And the blended model is a weighted average of the two, weighted by the amount of games the player played. Simply put, the less games the player played, the more the blended model tends to the global model.

### - Models are trained on historical rolling averages + opponent metrics

### - Model optimization
        - To optimize the model parameters press "tune hyperparameters". You will see the studies running in the backend terminal. After the studies finish, the models will retrain. You will not be able to make a prediction during this process (usually 1-2 minutes). Similarly, press "Train Global Model" to retrain the global model, if data has been added.

### - Run Saving
        - Press "Save Run" if you want to be able to look back at this specific player prediciton's run metrics, including mae, rmse, r2, bias, within_n, and feature importance. This is accessible in the "Run History" tab.

## - Metrics
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R^2
    - Bias
    - Within_n: 
        - This displays the % of time that the predicted value is "within +/- n points" of the actual value. n is 3 by default, but can be changed in constants.py.


### Significance
So what's the significance of the results? Well for now:
- Predicted Points is the Predicted Points total for your player, against the opponent you selected.
- What about MAE/RMSE? Simply put, these are model metrics, NOT affected by the single game and opponent chosen.
- The MAE (mean average error) value is from the model's training. So, for example, an MAE of 2.3 indicates that on average, the model in testing was off by 2.3 points in its prediction.
- The RMSE (root mean squared error) value is from the model's training. However, it punishes large outliers more than MAE does. So, for example, an RMSE of 2.3 indicates that on average, the model in testing was off by 2.3 points in its prediction.

---

## Notes
Ensure player data is stored in:
```
backend/data/player_game_data/{TEAM}/{PLAYER}.csv
```
And each CSV contains full box score stats per game.

### Right now, there is all 32 teams defensive stats. However, There is a limited amount of players to train on, which will be expanded on.
---

Enjoy predicting like a pro!

## 📊 Data Source

Player and team statistics used in this project were sourced from (nba.com) using nba_api.



## 🪪 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

You are free to use, modify, and distribute this software, provided that any derivative work is also released under the same license.  
**Commercial use is allowed**, but **derivatives must also be open source** under GPL.

If you're unsure about using this project in a commercial setting, please reach out.
