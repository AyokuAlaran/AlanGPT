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
        model = joblib.load('production_model.pkl')
        matches = pd.read_parquet('processed_matches.parquet')
        return model, matches
    except FileNotFoundError:
        st.error("🚨 Missing Files! Please upload 'production_model.pkl' and 'processed_matches.parquet' to GitHub.")
        st.stop()

# ==========================================
# 3. PREDICTION ENGINE (UPDATED FOR NEW COLUMNS)
# ==========================================
def run_ai_prediction(team_a, team_b, model, matches_df):
    
    def get_stats(team):
        # 1. Find the last match this team played
        last = matches_df[(matches_df['home_team'] == team) | (matches_df['away_team'] == team)].iloc[-1]
        is_t1 = last['home_team'] == team
        
        # 2. Extract the NEW column names (recent vs season)
        return {
            "skill": last['t1_skill'] if is_t1 else last['t2_skill'],
            "recent_op": last['t1_recent_op'] if is_t1 else last['t2_recent_op'],
            "recent_ds": last['t1_recent_ds'] if is_t1 else last['t2_recent_ds'],
            "season_op": last['t1_season_op'] if is_t1 else last['t2_season_op'],
            "season_ds": last['t1_season_ds'] if is_t1 else last['t2_season_ds'],
            "op_diff": last['t1_op_diff'] if is_t1 else last['t2_op_diff']
        }

    s_a = get_stats(team_a)
    s_b = get_stats(team_b)

    # 3. Prepare Input for XGBoost (Must match Training Order EXACTLY)
    # The model expects these 11 specific columns now
    features = [
        s_a['skill'] - s_b['skill'], # skill_gap
        s_a['recent_op'], s_a['recent_ds'], s_a['season_op'], s_a['season_ds'], s_a['op_diff'],
        s_b['recent_op'], s_b['recent_ds'], s_b['season_op'], s_b['season_ds'], s_b['op_diff']
    ]
    
    input_data = pd.DataFrame([features], columns=[
        'skill_gap', 
        't1_recent_op', 't1_recent_ds', 't1_season_op', 't1_season_ds', 't1_op_diff',
        't2_recent_op', 't2_recent_ds', 't2_season_op', 't2_season_ds', 't2_op_diff'
    ])
    
    # 4. Predict
    raw_probs = model.predict_proba(input_data)[0]
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
