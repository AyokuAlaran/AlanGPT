import streamlit as st
import pandas as pd
import os
import joblib

# 1. SETUP PAGE
st.set_page_config(page_title="BPCL Scout (Debug)", page_icon="🔧")

st.title("🔧 Diagnostic Mode")
st.write("If you see this, the app has started.")

# 2. CHECK FILES (Debugging step)
st.write("---")
st.write("📂 **Checking File System...**")

files_needed = ['production_model.pkl', 'production_scaler.pkl', 'processed_matches.parquet']
missing_files = []

for f in files_needed:
    if os.path.exists(f):
        st.success(f"✅ Found: {f}")
    else:
        st.error(f"❌ MISSING: {f}")
        missing_files.append(f)

if missing_files:
    st.error("🚨 CRITICAL ERROR: You must upload the missing files to GitHub!")
    st.stop() # Stops here if files are missing

# 3. ATTEMPT IMPORT (Debugging step)
st.write("---")
st.write("🔄 **Importing Backend...**")
try:
    import gemini_backend_v6 as backend
    st.success("✅ Backend Imported Successfully")
except Exception as e:
    st.error(f"❌ Backend Import Failed: {e}")
    st.stop()

# 4. ATTEMPT LOAD (Debugging step)
st.write("---")
st.write("🧠 **Loading Brain (Model & Scaler)...**")

# We deliberately DO NOT use cache here to force a fresh load and see errors
try:
    model, scaler, matches = backend.load_production_system()
    st.success("✅ System Loaded! Model, Scaler, and Data are ready.")
except ValueError as e:
    st.error(f"❌ Unpacking Error: {e}")
    st.warning("Hint: Did you update the backend to return 3 items (model, scaler, matches)?")
    st.stop()
except Exception as e:
    st.error(f"❌ Loading Error: {e}")
    st.stop()

# 5. UI RENDER (If we get here, the app works)
st.write("---")
st.success("🚀 UI RENDER STARTING...")

teams = sorted(matches['home_team'].unique())
t1 = st.selectbox("Test Select 1", teams)
t2 = st.selectbox("Test Select 2", teams, index=1)

if st.button("Test Generate"):
    st.write("Button Clicked!")
