import streamlit as st
import pandas as pd
import os
import gemini_backend_v6 as backend

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (Teal/White Theme)
# ==========================================
st.set_page_config(
    page_title="BPCL Scout | Alan Corp.",
    page_icon="⚽",
    layout="centered"
)

# Custom CSS for Centering, Buttons, and Layout
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #F5F9F9;
        color: #0F3D3D;
    }
    
    /* Centered Titles */
    .main-title {
        text-align: center;
        color: #008080 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Tactile Buttons */
    div.stButton > button {
        background-color: #008080;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.1s ease-in-out;
        width: 100%;
    }
    div.stButton > button:active {
        transform: translateY(2px);
        box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.2);
        background-color: #006666;
    }

    /* Result Card */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.08);
        margin-top: 20px;
        border-top: 5px solid #008080;
        color: #333;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #888;
        font-size: 12px;
        font-family: monospace;
        border-top: 1px solid #eee;
        padding-top: 20px;
    }
    .footer a {
        color: #008080;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SMART LOGO FINDER
# ==========================================
def get_logo_path(team_name):
    # Check all possible extensions
    extensions = ["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]
    base_folder = "team_logos"
    
    for ext in extensions:
        # Construct path: team_logos/TeamName.png
        file_path = f"{base_folder}/{team_name}.{ext}"
        # Check if file exists (works locally and often on cloud if folder structure is right)
        if os.path.exists(file_path):
            return file_path
            
    # Fallback: If you are on Streamlit Cloud, os.path.exists might be tricky 
    # if the folder isn't at root. We return the most common one to try.
    return f"{base_folder}/{team_name}.png"

# ==========================================
# 3. BACKEND LOADER
# ==========================================
@st.cache_resource
def load_engine():
    model, processed_data = backend.preprocess_and_train()
    return model, processed_data

# ==========================================
# OPTIMIZED LOADING
# ==========================================
@st.cache_resource
def load_engine():
    # Loads the files ONCE per server reboot. Fast.
    return backend.load_production_system()

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_report(t1, t2):
    # Caches the Gemini Result for 1 hour.
    # If User A generates this, User B sees it instantly.
    return backend.run_ai_prediction(t1, t2, model, matches)

# HEADER
st.link_button("🏆 View League Table", "https://myproclubs.com/t/bpcl")
st.markdown('<h1 class="main-title">BPCL Match Predictor</h1>', unsafe_allow_html=True)

# LOAD
with st.spinner("Loading System..."):
    model, matches = load_engine()
    team_list = sorted(matches['home_team'].unique())

# INPUTS
c1, c2 = st.columns(2)
with c1: team_a = st.selectbox("Home Team", team_list, key="t1")
with c2: team_b = st.selectbox("Away Team", team_list, index=1, key="t2")

# LOGIC
if st.button("GENERATE SCOUT REPORT"):
    if team_a != team_b:
        with st.spinner(f"🤖 Analyzing {team_a} vs {team_b}..."):
            
            # CALL CACHED FUNCTION
            raw_text = get_cached_report(team_a, team_b)
            
            # Parse
            try:
                parts = raw_text.split("###")
                percents = parts[1].replace("PERCENTS", "").strip() if len(parts)>1 else "Pending"
                insight = parts[2].replace("INSIGHT", "").strip() if len(parts)>2 else raw_text
                reasoning = parts[3].replace("REASONING", "").strip() if len(parts)>3 else "..."
            except:
                percents, insight, reasoning = "Error", raw_text, "..."

            # Display
            st.write("---")
            f1, f2, f3 = st.columns([1, 2, 1])
            with f1: 
                if os.path.exists(get_logo_path(team_a)): st.image(get_logo_path(team_a), width=80)
                else: st.write(f"**{team_a}**")
            with f2: st.markdown(f'<div class="fixture-score">{percents}</div>', unsafe_allow_html=True)
            with f3: 
                if os.path.exists(get_logo_path(team_b)): st.image(get_logo_path(team_b), width=80)
                else: st.write(f"**{team_b}**")
            
            st.markdown(f'<div class="result-card"><h3>📢 Tactical Insight</h3>{insight}</div>', unsafe_allow_html=True)
            with st.expander("Show Model Reasoning"): st.info(reasoning)
