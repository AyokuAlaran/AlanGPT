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
    SYSTEM: You are the BPCL Tactical Analyst.
    CONTEXT: All matches are played at a NEUTRAL VENUE. There is NO home advantage or away disadvantage. Treat both teams as playing on level ground.
    MATCH: {team_a} vs {team_b}
    
    RAW XGBOOST FORECAST (For Reference):
    - {team_a} Win: {smooth_probs[2]*100:.1f}%
    - Draw: {smooth_probs[1]*100:.1f}%
    - {team_b} Win: {smooth_probs[0]*100:.1f}%
    
    FORM GUIDE (Last 4 Games):
    - {team_a}: Averaging {s_a['op']:.1f} goals scored/game
    - {team_b}: Averaging {s_b['op']:.1f} goals scored/game

    YOUR TASKS:
    1. ANALYZE: Identify if XGBoost is overreacting to small sample sizes (e.g., if a team has one lucky 6-0 win but low skill, the math might be too high).
    2. CALIBRATE: Adjust the raw percentages based on your analysis of the sample size and neutral venue.
    3. PREVIEW: Write a 'Tactical Preview' (Strictly Max 3 sentences) explaining the clash.
    
    OUTPUT FORMAT:
    ### PERCENTS
    [Your Final Calibrated Percentages: Home% - Draw% - Away%]
    
    ### INSIGHT
    [The Tactical Preview: Max 3 sentences. Focus on style of play and the stats that affected the prediction.]
    
    ### REASONING
    [Explain your calibration. Explicitly mention if you adjusted for 'Overreaction to Form (Momentum)' or 'Small Sample Size'.]
    """

    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text
