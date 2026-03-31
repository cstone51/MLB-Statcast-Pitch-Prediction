

# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer



# #features = ['balls', 'strikes', 'inning', 'outs_when_up','inning_top','pitch_type_map', 'batter_is_right', 'runner_on_first']
# def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
#     # df = df.copy()
    
#     # #adds numericmapping for pitch types
#     # pitch_dict = {
#     #     'FF':0, 'FA':0, 'FT':0,
#     #     'SI':1, 
#     #     'FC': 5,
#     #     'CU':2,'KC':2,'CS':2,'EP':2,
#     #     'SL':3, 'ST': 3,
#     #     'CH':4,'FS':4,'FO':4,'SC':4,
#     #     'KN':6, 'GY':6, 
#     #     'PO':np.nan}
    
#     # df['pitch_type_map'] = df['pitch_type'].map(pitch_dict)

#     df['on_1b'] = df['on_1b'] .fillna(0)
#     df['runner_on_first'] = np.where(df['on_1b'] > 0.0001, 1, 0)

#     df['prev_pitch_1'] = df['pitch_type'].shift(1)
#     df['prev_pitch_2'] = df['pitch_type'].shift(2)
    
#     df['inning_top'] = np.where(df['inning_topbot'] == 'Top', 1, 0)

#     df['batter_is_right'] = np.where(df['stand'] == 'R', 1, 0)

#     return df

# NUMERIC_FEATURES = ["balls", "strikes", "inning", "outs_when_up"]
# CATEGORICAL_FEATURES = ["inning_top","batter_is_right", "runner_on_first"]


# def build_feature_pipeline() -> ColumnTransformer:
#     """Return a sklearn ColumnTransformer for preprocessing."""
#     numeric_pipe = Pipeline([
#         ("scaler", StandardScaler()),
#     ])
#     categorical_pipe = Pipeline([
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
#     ])
#     return ColumnTransformer([
#         ("num", numeric_pipe, NUMERIC_FEATURES),
#         ("cat", categorical_pipe, CATEGORICAL_FEATURES),
#     ])


# def build_features(df: pd.DataFrame):
#     """Convenience wrapper — returns feature matrix X and target y."""
#     X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
#     y = df["pitch_type"]
#     return X, y

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df['on_1b'] = df['on_1b'].fillna(0)
    df['runner_on_first'] = np.where(df['on_1b'] > 0.0001, 1, 0)

    df['inning_top'] = np.where(df['inning_topbot'] == 'Top', 1, 0)
    df['batter_is_right'] = np.where(df['stand'] == 'R', 1, 0)

    # Shift within at-bat only — first pitch of each at-bat gets NaN, not
    # the last pitch of the previous at-bat
    df['prev_pitch_1'] = (
        df.groupby('at_bat_number')['pitch_type']
        .shift(1)
        .fillna('NONE')   # 'NONE' = "this is the first pitch of the at-bat"
    )
    df['prev_pitch_2'] = (
        df.groupby('at_bat_number')['pitch_type']
        .shift(2)
        .fillna('NONE')   # 'NONE' = "no second previous pitch exists"
    )

    return df


NUMERIC_FEATURES     = ["balls", "strikes", "inning", "outs_when_up"]
CATEGORICAL_FEATURES = ["inning_top", "batter_is_right", "runner_on_first",
                        "prev_pitch_1", "prev_pitch_2"]


def build_feature_pipeline() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        # handle_unknown="ignore" means unseen pitch types at inference
        # (e.g. a pitch type in test not seen in train) get all-zero encoding
        # rather than crashing
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_features(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["pitch_type"]
    return X, y