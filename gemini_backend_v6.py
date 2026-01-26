%%writefile gemini_backend_v6.py
import streamlit as st
import pandas as pd
import joblib
from google import genai

# ==========================================
# 1. SETUP & AUTH
# ==========================================
try:
    # Check secrets first
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        # Local fallback
        API_KEY = "YOUR_RAW_KEY"
    
    client = genai.Client(api_key=API_KEY)
    MODEL_ID = 'gemini-2.5-flash' 
except Exception as e:
    pass # Frontend handles errors

# ==========================================
# 2. FAST LOADER (Reads the uploaded files)
# ==========================================
def load_production_system():
    try:
        # Instant load (milliseconds)
        model = joblib.load('production_model.pkl')
        matches = pd.read_parquet('processed_matches.parquet')
        return model, matches
    except FileNotFoundError:
        st.error("🚨 System Files Missing! Please upload 'production_model.pkl' and 'processed_matches.parquet' to GitHub.")
        st.stop()

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
def run_ai_prediction(team_a, team_b, model, matches_df):
    # Fetch latest stats
    def get_latest(team):
        # Find the last game this team played (home or away)
        last = matches_df[(matches_df['home_team'] == team) | (matches_df['away_team'] == team)].iloc[-1]
        is_t1 = last['home_team'] == team
        # Extract the symmetric stats we calculated during training
        return {
            "skill": last['t1_skill'] if is_t1 else last['t2_skill'],
            "op": last['t1_op'] if is_t1 else last['t2_op'],
            "ds": last['t1_ds'] if is_t1 else last['t2_ds']
        }

    s_a = get_latest(team_a)
    s_b = get_latest(team_b)

    # Prepare input vector for XGBoost
    # Note: We must match the columns used during training EXACTLY
    input_data = pd.DataFrame([[
        s_a['skill']-s_b['skill'], # skill_gap
        s_a['op'], s_a['ds'],      # t1 form
        s_b['op'], s_b['ds']       # t2 form
    ]], columns=['skill_gap', 't1_op', 't1_ds', 't2_op', 't2_ds'])
    
    # Get Math Probabilities
    raw_probs = model.predict_proba(input_data)[0]
    
    # Bayesian Smoothing (Prevent 99% confidence)
    smooth_probs = (raw_probs * 0.85) + (0.15 / 3)

    # Prompt for Gemini
    prompt = f"""
    SYSTEM: BPCL Tactical Analyst.
    MATCH: {team_a} vs {team_b}
    
    MATH FORECAST:
    - {team_a} Win: {smooth_probs[2]*100:.1f}%
    - Draw: {smooth_probs[1]*100:.1f}%
    - {team_b} Win: {smooth_probs[0]*100:.1f}%
    
    FORM GUIDE:
    - {team_a}: {s_a['op']:.1f} goals/game
    - {team_b}: {s_b['op']:.1f} goals/game

    OUTPUT FORMAT:
    ### PERCENTS
    {smooth_probs[2]*100:.0f}% - {smooth_probs[1]*100:.0f}% - {smooth_probs[0]*100:.0f}%
    
    ### INSIGHT
    [3 sentences. Tactical focus. No numbers.]
    
    ### REASONING
    [Explain why the model favors one team. Discuss Skill Gap vs Momentum.]
    """

    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text
