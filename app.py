import streamlit as st
import pandas as pd
import os
import gemini_backend_v5 as backend

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
# 4. HEADER SECTION (Button + Titles)
# ==========================================
# Top row for the external link button
col_spacer, col_btn = st.columns([3, 1])
with col_btn:
    st.link_button("View League Table", "https://myproclubs.com/t/bpcl")

# Centered Headings
st.markdown('<h1 class="main-title">BPCL Match Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Tactical Scouting</div>', unsafe_allow_html=True)

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
# Load Brain
with st.spinner("Initializing Neural Engine..."):
    model, matches = load_engine()
    team_list = sorted(matches['home_team'].unique())

# Team Selection
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Select Home Team", team_list, key="t1")
    # Try to show logo
    logo_a = get_logo_path(team_a)
    if os.path.exists(logo_a):
        st.image(logo_a, width=120)

with col2:
    team_b = st.selectbox("Select Away Team", team_list, index=1, key="t2")
    logo_b = get_logo_path(team_b)
    if os.path.exists(logo_b):
        st.image(logo_b, width=120)

st.write("") # Spacer

# ==========================================
# 6. PREDICTION ENGINE
# ==========================================
if st.button("GENERATE SCOUT REPORT"):
    if team_a == team_b:
        st.warning("⚠️ Please select two different teams.")
    else:
        # REAL LOADER: Stays active until backend finishes
        with st.spinner(f"Crunching numbers for {team_a} vs {team_b}..."):
            
            # 1. Get raw text from Gemini
            raw_report = backend.run_ai_prediction(team_a, team_b, model, matches)
            
            # 2. Clean Asterisks (**)
            clean_report = raw_report.replace("**", "")
            
            # 3. Display
            st.markdown(f"""
            <div class="result-card">
                <h3>📊 Tactical Analysis</h3>
                {clean_report.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    Built by <b>Alan Corp.</b><br>
    Contact: <a href="mailto:aelanlockin@gmail.com">aelanlockin@gmail.com</a>
</div>
""", unsafe_allow_html=True)
