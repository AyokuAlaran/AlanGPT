import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google import genai
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback for when you run it locally on your own machine
    API_KEY = "YOUR_ACTUAL_RAW_KEY_HERE"

client = genai.Client(api_key=API_KEY)
gemini_model = 'gemini-2.5-flash'
@st.cache_resource
def load_production_system():
    # Load the pre-trained brain
    model = joblib.load('production_model.pkl')
    # Load the data using Parquet (Fast!)
    matches = pd.read_parquet('processed_matches.parquet')
    return model, matches
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
