# NBA Player Points Predictor

This is a full-stack web app that predicts NBA player points based on past game performance and opponent defense metrics. Part originally of a stats project in school, but something I thought I'd build out some UI for.

It's a great chance to improve and look into modeling, machine learning, and potentially neural networks in the future.

## Folder Structure

```
player-predictor-app/
├── backend/
│   ├── main.py                  # FastAPI entrypoint
│   ├── model_logic.py           # Model logic and data loading
│   ├── player_data_setup.py     # Provided preprocessing functions
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
│   │   ├── App.jsx              # Main React UI
│   │   ├── index.js             # React DOM root
│   │   └── index.css            # Tailwind styles
│
├── README.md
```

---

## 🚀 How to Run

### Backend
```bash
cd player_predition_app/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install fastapi uvicorn scikit-learn pandas numpy
uvicorn main:app --reload
```

### Frontend
```bash
cd player_prediction_app/frontend
npm install
npm run dev
```

Frontend will start on `http://localhost:5173` and communicate with FastAPI at `http://localhost:8000`.

---

## ✅ Features
- Select your player, team, and opponent for game prediction
- Run prediction and view predicted single game points, MAE, RMSE
- Trained on historical rolling averages + opponent metrics

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

Player and team statistics used in this project were sourced from [Basketball-Reference.com](https://www.basketball-reference.com/).

Basketball Reference is a fantastic resource for historical NBA data and analytics.  
This project is not affiliated with or endorsed by Basketball Reference or Sports Reference LLC.


## 🪪 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

You are free to use, modify, and distribute this software, provided that any derivative work is also released under the same license.  
**Commercial use is allowed**, but **derivatives must also be open source** under GPL.

If you're unsure about using this project in a commercial setting, please reach out.
