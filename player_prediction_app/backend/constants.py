''' Constants throughout the project'''

# Column to predict
TARGET_COLUMN = "PTS"

# Rolling window to use for data SETUP 
ROLLING_TREND_WINDOW = 5

# Rolling window to use for input row creation
ROLLING_WINDOW = 10

# Columns to train the model with
FEATURE_COLUMNS = [
    "Home", "Pace", "eFG%_y", "TOV%", "DRB%",
    "PTS_last_5_avg", "MP_last_5_avg", "PTS_trend_5",
    "PTS_vol_5", "PTS_per_min", "def_adj",
]

# Columns to Scale (must be in FEATURE_COLUMNS too)
COLUMNS_TO_SCALE = [  "Pace",
        'PTS_last_5_avg',  'MP_last_5_avg',
            # 'PTS_5_game_trend',
            'PTS_vol_5',
            # 'Hot_Streak',
            # 'PTS_rolling_trend',
            'PTS_per_min',
            # 'PTS_pct_of_max', 
            'def_adj',
            'PTS_trend_5',
            #'Days_of_rest'
            
    ]

POTENTIAL_ONES_TO_TEST = ["MP_x_FGA",'FGA_last_5_avg', 'Opp_DRtg_x_PTS',
    'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS','DRtg', 'FT/FGA', "PTS_pct_of_max", "PTS_rolling_trend",'Hot_Streak',  "Days_of_rest"] # non-useful (for now) terms

IGNORED_COLUMNS = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%_x', "2P", "2PA", "2P%",
    ]