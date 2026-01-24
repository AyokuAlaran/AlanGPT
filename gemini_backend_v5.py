import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from google import genai
from sklearn.model_selection import train_test_split
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback for when you run it locally on your own machine
    API_KEY = "YOUR_ACTUAL_RAW_KEY_HERE"

client = genai.Client(api_key=API_KEY)
gemini_model = 'gemini-2.5-flash'
def preprocess_and_train():
    print("🏟️ Initializing Neutral Venue Pipeline...")

    # Load raw data
    matches = pd.read_csv('match_log.csv')
    attr = pd.read_csv('team_attributes.csv')

    # Date normalization
    matches['date'] = pd.to_datetime(matches['date'], dayfirst=True)
    attr['effective_from'] = pd.to_datetime(attr['effective_from'], dayfirst=True)

    # A. Mapping Skill (Neutral Context)
    def get_skill(team, match_date):
        h = attr[(attr['team_name'] == team) & (attr['effective_from'] <= match_date)]
        return h.iloc[-1]['skill_rating'] if not h.empty else 1500

    matches['t1_skill'] = matches.apply(lambda x: get_skill(x['home_team'], x['date']), axis=1)
    matches['t2_skill'] = matches.apply(lambda x: get_skill(x['away_team'], x['date']), axis=1)

    # B. Symmetric Rolling Stats (OP/DS)
    # Treat every match from both perspectives
    t1 = matches[['home_team', 'home_score', 'away_score', 'date']].rename(columns={'home_team':'t', 'home_score':'gs', 'away_score':'gc'})
    t2 = matches[['away_team', 'away_score', 'home_score', 'date']].rename(columns={'away_team':'t', 'away_score':'gs', 'home_score':'gc'})
    timeline = pd.concat([t1, t2]).sort_values(['t', 'date'])

    timeline['op'] = timeline.groupby('t')['gs'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    timeline['ds'] = timeline.groupby('t')['gc'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    matches = matches.merge(timeline[['t', 'date', 'op', 'ds']], left_on=['home_team', 'date'], right_on=['t', 'date'], how='left').rename(columns={'op':'t1_op', 'ds':'t1_ds'}).drop(columns='t')
    matches = matches.merge(timeline[['t', 'date', 'op', 'ds']], left_on=['away_team', 'date'], right_on=['t', 'date'], how='left').rename(columns={'op':'t2_op', 'ds':'t2_ds'}).drop(columns='t')
    matches.fillna(0, inplace=True)

    # C. Data Augmentation for Zero Bias
    # We swap Team 1 and Team 2 to create a symmetric training set
    swapped = matches.copy()
    swapped[['home_team', 'away_team']] = matches[['away_team', 'home_team']]
    swapped[['t1_skill', 't2_skill']] = matches[['t2_skill', 't1_skill']]
    swapped[['t1_op', 't2_op']] = matches[['t2_op', 't1_op']]
    swapped[['t1_ds', 't2_ds']] = matches[['t2_ds', 't1_ds']]

    # Map results: Win(2)->Loss(0), Loss(0)->Win(2), Draw(1)->Draw(1)
    matches['result'] = np.where(matches['home_score'] > matches['away_score'], 2, np.where(matches['home_score'] < matches['away_score'], 0, 1))
    swapped['result'] = matches['result'].map({2:0, 0:2, 1:1})

    final_train = pd.concat([matches, swapped])
    final_train['skill_gap'] = final_train['t1_skill'] - final_train['t2_skill']

    # D. Train Neutral XGBoost
    features = ['t1_skill', 't2_skill', 'skill_gap', 't1_op', 't1_ds', 't2_op', 't2_ds']
    model = xgb.XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, reg_lambda=15)
    model.fit(final_train[features], final_train['result'])

    joblib.dump(model, 'neutral_bpcl_model.pkl')
    print("✅ Model trained for Neutral Venue (Bias Zeroed).")
    return model, final_train
def run_ai_prediction(team_a, team_b, model, matches_df):
    def get_latest(team):
        last = matches_df[(matches_df['home_team'] == team) | (matches_df['away_team'] == team)].iloc[-1]
        is_t1 = last['home_team'] == team
        return {
            "skill": last['t1_skill'] if is_t1 else last['t2_skill'],
            "op": last['t1_op'] if is_t1 else last['t2_op'],
            "ds": last['t1_ds'] if is_t1 else last['t2_ds']
        }

    s_a = get_latest(team_a)
    s_b = get_latest(team_b)

    # Raw Math
    input_data = pd.DataFrame([[s_a['skill'], s_b['skill'], s_a['skill']-s_b['skill'], s_a['op'], s_a['ds'], s_b['op'], s_b['ds']]],
                               columns=['t1_skill', 't2_skill', 'skill_gap', 't1_op', 't1_ds', 't2_op', 't2_ds'])

    raw_probs = model.predict_proba(input_data)[0]

    # Bayesian Smoothing for Neutral Grounds (80% confidence in math, 20% flat variance)
    smooth_probs = (raw_probs * 0.8) + (0.2 / 3)

    prompt = f"""
    SYSTEM: BPCL Winter '26 Tactical Analyst. Venue: Neutral Grounds.
    MATCHUP: {team_a} vs {team_b}

    BASELINE DATA:
    - {team_a}: Skill {int(s_a['skill'])}, Form {s_a['op']:.1f} GS/game.
    - {team_b}: Skill {int(s_b['skill'])}, Form {s_b['op']:.1f} GS/game.

    XGBOOST PROBABILITIES (Symmetric):
    {team_a} Win: {smooth_probs[2]*100:.1f}%, Draw: {smooth_probs[1]*100:.1f}%, {team_b} Win: {smooth_probs[0]*100:.1f}%

    TASK:
    1. Identify if XGBoost is overreacting to small sample sizes (e.g., a single 6-0 win).
    2. Note: Neglect Home-advantage/Away-disadvantages effects and the resulting bias; all the matches are played in a neutral venue, without favoritism to either team.
    3. Provide your 'Calibrated Percentages' for the match.
    4. Write a 'Tactical Preview' (max 3 sentences) explaining the clash.
    """

    response = client.models.generate_content(
                        model=gemini_model,
                        contents=prompt)
    return response.text
if __name__ == "__main__":
    trained_model, processed_data = preprocess_and_train()
