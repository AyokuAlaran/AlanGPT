import streamlit as st
import pandas as pd
import os
import gemini_backend_v6 as backend

# 1. CONFIG
st.set_page_config(page_title="BPCL Scout", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #F5F9F9; color: #0F3D3D; }
    .main-title { text-align: center; color: #008080 !important; font-size: 3rem; margin-bottom: 0px; font-weight: 800; }
    .fixture-score { 
        text-align: center; font-size: 20px; font-weight: 900; 
        color: #008080; background: white; padding: 15px; 
        border-radius: 12px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0f2f1;
    }
    .result-card { background-color: white; padding: 25px; border-radius: 15px; border-top: 5px solid #008080; margin-top: 20px;}
    div.stButton > button { 
        background-color: #008080; color: white; border-radius: 12px; 
        padding: 12px; font-size: 18px; font-weight: bold; width: 100%; border: none;
    }
</style>
""", unsafe_allow_html=True)

# 2. LOGO UTILS
def get_logo_path(team_name):
    for ext in ["png", "PNG", "jpg", "JPG"]:
        path = f"team_logos/{team_name}.{ext}"
        if os.path.exists(path): return path
    return None

# 3. HIGH SPEED LOADING
@st.cache_resource
def load_system():
    # Only loads once. Super fast.
    return backend.load_production_system()

@st.cache_data(ttl=3600, show_spinner=False)
def get_report(t1, t2):
    return backend.run_ai_prediction(t1, t2, model, matches)

# 4. UI LAYOUT
st.link_button("🏆 League Table", "https://myproclubs.com/t/bpcl")
st.markdown('<h1 class="main-title">BPCL Match Predictor</h1>', unsafe_allow_html=True)

with st.spinner("Booting AI..."):
    model, matches = load_system()
    teams = sorted(matches['home_team'].unique())

c1, c2 = st.columns(2)
with c1: t1 = st.selectbox("Home", teams, key="1")
with c2: t2 = st.selectbox("Away", teams, index=1, key="2")

# 5. PREDICTION
if st.button("GENERATE SCOUT REPORT"):
    if t1 != t2:
        with st.spinner(f"⚡ Analyzing {t1} vs {t2}..."):
            raw = get_report(t1, t2)
            
            # Parsing
            try:
                parts = raw.split("###")
                percents = parts[1].replace("PERCENTS", "").strip()
                insight = parts[2].replace("INSIGHT", "").strip()
                reasoning = parts[3].replace("REASONING", "").strip()
            except:
                percents, insight, reasoning = "Pending", raw, "Analysis Error"

            # FIXTURE DISPLAY
            st.write("---")
            c_a, c_mid, c_b = st.columns([1, 2, 1])
            with c_a:
                logo = get_logo_path(t1)
                if logo: st.image(logo, width=90)
                else: st.markdown(f"**{t1}**")
            
            with c_mid:
                st.markdown(f"<div class='fixture-score'>{percents}</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; font-size:12px; color:#888;'>Win % - Draw % - Win %</div>", unsafe_allow_html=True)

            with c_b:
                logo = get_logo_path(t2)
                if logo: st.image(logo, width=90)
                else: st.markdown(f"**{t2}**")

            st.write("---")
            
            # INSIGHT
            st.markdown(f"<div class='result-card'><h3>📢 Tactical Insight</h3>{insight}</div>", unsafe_allow_html=True)
            
            # HIDDEN REASONING
            st.write("")
            with st.expander("Show Model Reasoning (Technical Data)"):
                st.info(reasoning)
