from pybaseball import  playerid_lookup # type: ignore
from pybaseball import  statcast_pitcher # type: ignore
from pybaseball import  statcast # type: ignore
import pandas as pd
import numpy as np

def pull_data (first_name, last_name, start_date, end_date, team_abbreviation = None):
    
    '''
    This makes the call to the pybaseball function player_id lookup to obtain the id value for the pitcher
    
    If it detects multiple pitchers of the same name, if will select the most recent pitcher to debut
    
    Then it will use the id to obtain their stats in a given timeframe. In this case for the 2023 season
    '''
    try: 
        name = playerid_lookup(last_name, first_name)
        if len(name) > 1:
            name = name.sort_values(by='mlb_played_first', ascending=False)
            name = name.head(1)
        id = name['key_mlbam'].values[0]
        df = statcast_pitcher(start_date, end_date, id)
        return df
    except:
        print('Please check name spelling, or add team name abbreviation for more specificity')



def add_features (df):
    df = df.copy()
    '''
    This takes several steps to add lag features and other aggregate features that should be helpful in modeling
    
    '''
    pitch_dict = {
            'FF':0, 'FA':0, 'FT':0,
            'SI':1, 
            'FC': 5,
              'CU':2,'KC':2,'CS':2,'EP':2,
              'SL':3, 'ST': 3,
              'CH':4,'FS':4,'FO':4,'SC':4,
              'KN':6, 'GY':6, 
              
              'PO':np.nan}

    # Map old pitch types to new mapping
    df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
    df['pitch_id'] = df['game_pk'].astype(str) + "-" + df['at_bat_number'].astype(str) + "-" + df['pitch_number'].astype(str)
    # df['pa_id'] = str(df['game_pk']) + "-" + str(df['at_bat_number'])
    # df['pitch_id'] = str(df['pa_id']) + "-" + str(df['pitch_number'])
    df['pitch_type_map'] = df['pitch_type'].map(pitch_dict)
    fastball_maps = [0, 1]
    df['is_fastball'] = np.where(df['pitch_type_map'].isin(fastball_maps), 1, 0)
    #dropping rows with Nan
    
    
    #df['pitch_type_map'] = df['pitch_type_map'].astype(int)
    df.dropna(subset=['pitch_type_map'], inplace = True)

   
    #add lag pitches
    df['prev_pitch_1'] = df['pitch_type_map'].shift(1)
    df['prev_pitch_2'] = df['pitch_type_map'].shift(2)
    df['prev_pitch_3'] = df['pitch_type_map'].shift(3)

    df['prev_pitch_fb'] = df['is_fastball'].shift(1)
    df['prev_2_pitches_fb'] = np.where((df['prev_pitch_1'] == 0) & (df['prev_pitch_2'] == 0), 1, 0)
    

    df['on_1b'] = df['on_1b'] .fillna(0)
    df['on_2b'] = df['on_2b'] .fillna(0)
    df['on_3b'] = df['on_3b'] .fillna(0)

    #df['runners_on_base'] = df['on_1b'] + df['on_2b'] + df['on_3b']

    #df['runners_on_base'].value_counts()
    df['runner_on_first'] = np.where(df['on_1b'] > 0.0001, 1, 0)
    df['prev_pitch_was_fb'] = np.where(df['prev_pitch_1'] == 0, 1, 0)
    df['two_strike_count'] = np.where(df['strikes'] == 2, 1, 0)
    df['hitters_count'] = np.where((df['balls'] == 3) | (df['strikes'].isin([0,1])), 1, 0)
    df['last_two_pitches_same'] = np.where((df['prev_pitch_1'] == df['prev_pitch_2']) & (df['prev_pitch_1'].notna()), 1, 0)
   
   
    #current_pitch = row['pitch_id']
            
    df["fb_prop_last_10"] = (
        df["is_fastball"]
        .shift(1)               
        .rolling(15, min_periods=1)
        .mean()
        )
    
    df = pd.get_dummies(df, columns=['prev_pitch_1'], dtype=int)
    return df


# def remove_factors (df):
    
#     df = df.copy()
#     '''
#     This will remove columns with missing values, and also remove  features that occur during or after the pitch 
#     since we will only be interested in predicitive and situational features
    
#     '''
#     #missing values
#     missing = df.isna().sum().reset_index()
#     missing = missing.loc[missing[0] > 0]
#     removal_list = missing['index'].to_list()

    

#     df = df.drop(removal_list, axis=1)
#     return df
    
def encode(df):

    '''
    This transforms all remaaining columns with categorical data types to either binary or dummy variables
    '''
    df['batter_is_right'] = np.where(df['stand'] == 'R', 1, 0)
    
    df['pitcher_is_right'] = np.where(df['p_throws'] == 'R', 1, 0)
    df['inning_top'] = np.where(df['inning_topbot'] == 'Top', 1, 0)
    #df = df.drop(['stand', 'p_throws', 'inning_topbot'], axis=1)
    #df = pd.get_dummies(df, columns=['if_fielding_alignment', 'of_fielding_alignment'], drop_first=True)
    
    return df
        
        
def pull_pitcher_data(first_name, last_name, start_date, end_date):
    
    
    
    df = pull_data(first_name, last_name, start_date, end_date)
    df = add_features(df)
    #df = remove_factors(df)
    df = encode(df)
   
    #filtering to a final featureset
    
    # features = ['game_date', 'at_bat_number','pitch_type_map','balls', 'strikes', 'outs_when_up',
    #     'home_score_diff', 'runner_on_first', 'inning_top',
    #     'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3', 'fb_prop_last_10',
    #     'batter_is_right', 'pitcher_is_right', 'pitch_number', 'inning', 'n_thruorder_pitcher', 'prev_pitch_was_fb',
    #     'two_strike_count', 'hitters_count', 'last_two_pitches_same',  'prev_pitch_fb','prev_2_pitches_fb', 'is_fastball'
        
    #     ]
    


    #df = df.dropna()
    #df = df[features]
    #df = df.dropna()
    return df
    #prop_columns = [col for col in df.columns if col.startswith('prop')]
    

    
    