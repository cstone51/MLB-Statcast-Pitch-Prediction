import pandas as pd
from pathlib import Path

from pybaseball import  playerid_lookup # type: ignore
from pybaseball import  statcast_pitcher # type: ignore
from pybaseball import  statcast # type: ignore

# Define the schema your data must conform to


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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    clean the raw data and return a DataFrame ready for feature engineering.
    '''
    df = df.copy()

    df.dropna(subset=["pitch_type"], inplace=True)
    df.fillna(0, inplace=True)
    remove_types = ['IN', 'PO', 'NP', 'EP', 'AB', 'AS', 'UN', 'GY']
    df = df[~df['pitch_type'].isin(remove_types)]


    # pitch_dict = {
    #         'FF':'Fastball', 'FA':'Fastball', 'FT':'Fastball', 'CS':'Fastball',
    #         'SI':'Sinker', 
    #         'FC':'Cutter',
    #         'CU':'Curveball','KC':'Curveball',
    #         'SL':'Slider', 'ST': 'Slider',
    #         'CH':'Changeup',
    #         'KN':'Knuckle', 'GY':6, 
              
    #         'PO':np.nan}
    return df