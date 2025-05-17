''' Constants throughout the project'''

# Cutoff number of games for training data
CUTOFF_VALUE = 10

N_CUTOFF = 3

# Column to predict
TARGET_COLUMN = "PTS"

# Rolling window to use for data SETUP 
ROLLING_TREND_WINDOW = 5

# Rolling window to use for input row creation
ROLLING_WINDOW = 10

# Columns to train the model with
# FEATURE_COLUMNS = [
#     "Home", "Pace", "eFG%_y", "TOV%", "DRB%",
#     "PTS_last_5_avg", "MP_last_5_avg", "PTS_trend_5",
#     "PTS_vol_5", "PTS_per_min", 'def_adj', "Days_of_rest", 
#     'usage_rate', "month_sin", "month_cos", "is_back2back", #"Hot_Streak"
# ]
FEATURE_COLUMNS = [
    # Game context
    'Days_of_rest', 'is_back2back', 'month_sin', 'month_cos',

    # Rolling means (r3, r5, r10)
    'PTS_r3', 'PTS_r5', 'PTS_r10',
    'FGA_r3', 'FGA_r5', 'FGA_r10',
    '3PA_r3', '3PA_r5', '3PA_r10',
    'FTA_r3', 'FTA_r5', 'FTA_r10',
    'TOV_r3', 'TOV_r5', 'TOV_r10',
    'MP_r3', 'MP_r5', 'MP_r10',
    'usage_rate_r3', 'usage_rate_r5', 'usage_rate_r10',

    # Rolling std
    'PTS_std_r5', 'FGA_std_r5',

    # Interaction features
    'Opp_DRtg_x_PTSr5',
    'Opp_Pace_x_FGAr5',
    'Opp_eFG_x_PTSr5',

    # Trend & volatility
    'PTS_trend_5',
    'PTS_per_min',
    'Hot_Streak',
    'PTS_z',

    # Opponent-adjusted scoring
    'def_adj',
]

# Columns to Scale (must be in FEATURE_COLUMNS too)
# COLUMNS_TO_SCALE = [  "Pace",
#         'PTS_last_5_avg',  'MP_last_5_avg',

#             'PTS_vol_5',
#             # 'Hot_Streak',

#             'PTS_per_min',
#             # 'PTS_pct_of_max', 
#             'def_adj',
#             'PTS_trend_5',
#             'Days_of_rest',
#             'usage_rate'
            
#     ]
COLUMNS_TO_SCALE = [
    'Days_of_rest',

    # Rolling stats
    'PTS_r3', 'PTS_r5', 'PTS_r10',
    'FGA_r3', 'FGA_r5', 'FGA_r10',
    '3PA_r3', '3PA_r5', '3PA_r10',
    'FTA_r3', 'FTA_r5', 'FTA_r10',
    'TOV_r3', 'TOV_r5', 'TOV_r10',
    'MP_r3', 'MP_r5', 'MP_r10',
    'usage_rate_r3', 'usage_rate_r5', 'usage_rate_r10',

    # Volatility & trends
    'PTS_std_r5', 'FGA_std_r5',
    'PTS_trend_5',
    'PTS_per_min',
    'PTS_z',

    # Interaction & adjusted scoring
    'Opp_DRtg_x_PTSr5',
    'Opp_Pace_x_FGAr5',
    'Opp_eFG_x_PTSr5',
    'def_adj',
]

POTENTIAL_ONES_TO_TEST = ["MP_x_FGA",'FGA_last_5_avg', 'Opp_DRtg_x_PTS',
    'Opp_Pace_x_FGA', 'Opp_eFG_x_PTS','DRtg', 'FT/FGA', "PTS_pct_of_max", "PTS_rolling_trend",'Hot_Streak',  "Days_of_rest"] # non-useful (for now) terms

IGNORED_COLUMNS = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA',
       'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'eFG%_x', "2P", "2PA", "2P%",
    ]