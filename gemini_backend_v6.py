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
        scaler = joblib.load('production_scaler.pkl') # Load the Normalizer
        matches = pd.read_parquet('processed_matches.parquet')
        return model, scaler, matches
    except FileNotFoundError:
        st.stop()

def run_ai_prediction(team_a, team_b, model, scaler, matches_df):
    
    # 1. Fetch Raw Stats
    def get_raw_stats(team):
        last = matches_df[(matches_df['home_team'] == team) | (matches_df['away_team'] == team)].iloc[-1]
        is_t1 = last['home_team'] == team
        return {
            "skill": last['t1_skill'] if is_t1 else last['t2_skill'],
            "recent": last['t1_recent'] if is_t1 else last['t2_recent'],
            "season": last['t1_season'] if is_t1 else last['t2_season'],
            "def": last['t1_def'] if is_t1 else last['t2_def']
        }
    
    s_a = get_raw_stats(team_a)
    s_b = get_raw_stats(team_b)

    # 2. Calculate Gaps
    raw_gaps = pd.DataFrame([[
        s_a['skill'] - s_b['skill'],
        s_a['recent'] - s_b['recent'],
        s_a['season'] - s_b['season'],
        s_b['def'] - s_a['def'] # Note the swap for defense (Lower is better)
    ]], columns=['gap_skill', 'gap_recent', 'gap_season', 'gap_def'])

    # 3. Normalize (Scale to -1 to 1 range using saved scaler)
    norm_gaps = scaler.transform(raw_gaps) # Returns a numpy array

    # 4. Apply YOUR Weights
    # [0]=Skill, [1]=Recent, [2]=Season, [3]=Defense
    weighted_score = (
        (norm_gaps[0][0] * 0.15) + 
        (norm_gaps[0][1] * 0.35) + 
        (norm_gaps[0][2] * 0.25) + 
        (norm_gaps[0][3] * 0.25)
    )

    # 5. Predict
    # XGBoost now only sees your calculated 'Dominance Score'
    input_data = pd.DataFrame([[weighted_score]], columns=['weighted_dominance'])
    raw_probs = model.predict_proba(input_data)[0]
    
    # ... [Prompt Logic same as before] ...
    
    prompt = f"""
    SYSTEM: BPCL Analyst.
    MATCH: {team_a} vs {team_b}
    
    ALGORITHM SCORE (Weighted): {weighted_score:.2f} 
    (Positive = Favors {team_a}, Negative = Favors {team_b})
    
    WEIGHTS APPLIED:
    - Skill: 15%
    - Recent Form: 35%
    - Season Form: 25%
    - Defensive Stats: 25%
    
    FORECAST:
    - {team_a}: {raw_probs[2]*100:.1f}%
    - Draw: {raw_probs[1]*100:.1f}%
    - {team_b}: {raw_probs[0]*100:.1f}%

    OUTPUT: 
    ### PERCENTS
    [Home-Draw-Away]
    ### INSIGHT
    [3 sentences]
    ### REASONING
    [Explain how the high weighting on Recent Form (35%) influenced this specific result.]
    """
    
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text
