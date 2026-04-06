



import joblib
import pandas as pd
from pathlib import Path

model   = joblib.load("notebooks/models/model.joblib")
le      = joblib.load("notebooks/models/label_encoder.joblib")

def predict_pitch(game_state: dict) -> dict:
    """
    Given a game state, return predicted pitch type and 
    probability for each class.
    """
    df = pd.DataFrame([game_state])

    pred_enc   = model.predict(df)[0]
    proba      = model.predict_proba(df)[0]

    pred_label = le.inverse_transform([pred_enc])[0]

    # zip class names with their probabilities, sorted by most likely
    proba_dict = dict(
        sorted(
            zip(le.classes_, proba),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        "prediction":    pred_label,
        "probabilities": proba_dict
    }


# Chris Sale, specific game state example:
result = predict_pitch({
    "balls":          3,
    "strikes":        0,
    "inning":         2,
    "outs_when_up":   1,
    "inning_top":     1,          # 1 = top of inning
    "batter_is_right": 0,         # 1 = right-handed batter
    "runner_on_first": 0,         # no runner = 0, runner = 1
    "prev_pitch_1":   "SL",       # abbr for pitch type before current pitch
    "prev_pitch_2":   "SL",       # abbr for pitch type before prev_pitch_1
})

print(result)