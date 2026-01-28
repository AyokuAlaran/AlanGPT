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
    
    # 5. Gemini Prompt (UPDATED KEYS HERE)
    # The error was happening because the old code used s_a['op'] here.
    prompt = f"""
    SYSTEM: You are a world-class football Tactical Analyst. You are able to read in between the team stats to make 95% correct predictions. The games hold on a Neutral Venue.
    MATCH: {team_a} vs {team_b}
    
    MATH PREDICTION:
    - {team_a} Win: {smooth_probs[2]*100:.1f}%
    - Draw: {smooth_probs[1]*100:.1f}%
    - {team_b} Win: {smooth_probs[0]*100:.1f}%
    
    DEEP DIVE FORM (Contextual):
    - {team_a}: Recent Avg {s_a['recent_op']:.1f} goals (Season Avg: {s_a['season_op']:.1f})
    - {team_b}: Recent Avg {s_b['recent_op']:.1f} goals (Season Avg: {s_b['season_op']:.1f})
    
    TASKS:
    1. Compare Recent Form vs Season Avg. Identify if a team is "Overperforming" (Hot Streak) or "Underperforming" (Slump).
    2. Adjust prediction: If a weak team has high recent form but low season stats, treat it as a "Purple Patch" (volatile).
    3. Output Calibrated Percentages and a 3-sentence preview.

    OUTPUT FORMAT:
    ### PERCENTS
    [Home% - Draw% - Away%]
    
    ### INSIGHT
    [3 sentences. Discuss if teams are in a "purple patch" or a "slump".]
    
    ### REASONING
    [Explain the Season vs Recent comparison and how it affected your forecast.]
    """

    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text
