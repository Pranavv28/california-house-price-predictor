"""
app.py — Streamlit Web App
California House Price Predictor · Maincrafts Technology AI/ML Internship
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #5f6368; font-size: 1rem; margin-top: 0; margin-bottom: 1.5rem; }
    .metric-card {
        background: #e8f0fe; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        border-left: 4px solid #1a73e8;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a73e8; }
    .metric-label { font-size: 0.85rem; color: #5f6368; margin-top: 2px; }
    .prediction-box {
        background: linear-gradient(135deg, #0d47a1, #1a73e8);
        border-radius: 16px; padding: 2rem; text-align: center; color: white;
    }
    .pred-value { font-size: 3rem; font-weight: 800; color: #ffd600; }
    .pred-label { font-size: 1rem; opacity: 0.85; margin-top: 0.3rem; }
    .badge {
        display: inline-block; background: #e8f0fe; color: #1a73e8;
        border-radius: 999px; padding: 2px 12px; font-size: 0.78rem;
        font-weight: 600; margin: 2px;
    }
    hr { border: 1px solid #e8eaed; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("linear_regression_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["scaler"], bundle["features"]
    except FileNotFoundError:
        return None, None, None


model, scaler, features = load_model()

# ── Header ────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-header">🏠 California House Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Linear Regression · Maincrafts Technology AI/ML Internship · Task 1</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align:right; padding-top:1.2rem;">
        <span class="badge">scikit-learn</span>
        <span class="badge">Python</span><br>
        <span class="badge">Linear Regression</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar: Model Info ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.6634</div>
        <div class="metric-label">R² Score</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.4607</div>
        <div class="metric-label">MAE (×$100k)</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.5571</div>
        <div class="metric-label">RMSE (×$100k)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Trained on the **California Housing Dataset** (1990 U.S. Census).  
    - 20,640 records  
    - 8 features  
    - 80/20 train-test split  
    - StandardScaler preprocessing
    """)

    st.markdown("---")
    st.markdown("### 🔢 Load Demo Values")
    if st.button("📋 Fill Demo Sample", use_container_width=True):
        st.session_state["demo"] = True

# ── Main Layout ───────────────────────────────────────────────
if model is None:
    st.error("⚠️ `linear_regression_model.pkl` not found. Run the notebook first to generate it.")
    st.stop()

demo = st.session_state.get("demo", False)

st.markdown("### 🎛️ Enter House Features")
st.caption("Adjust the sliders or type values for each feature, then click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    med_inc    = st.slider("💰 Median Income ($10k units)",   0.5, 15.0, 8.0 if demo else 5.0, 0.1,
                           help="Median household income of the block group")
    house_age  = st.slider("🏗️ House Age (years)",            1, 52, 25 if demo else 20,
                           help="Median age of houses in the block group")
    ave_rooms  = st.slider("🛋️ Average Rooms per Household",  1.0, 14.0, 6.0 if demo else 5.0, 0.1)
    ave_bedrms = st.slider("🛏️ Average Bedrooms per Household", 0.5, 3.4, 1.0 if demo else 1.0, 0.1)

with col2:
    population = st.number_input("👥 Block Group Population", 3, 35682, 500 if demo else 1200,
                                  help="Total population of the block group")
    ave_occup  = st.slider("🏘️ Average Household Occupancy",  1.0, 10.0, 2.5 if demo else 3.0, 0.1)
    latitude   = st.slider("🌐 Latitude (°N)",                32.5, 42.0, 37.0 if demo else 36.0, 0.01)
    longitude  = st.slider("🌐 Longitude (°W, enter negative)", -124.4, -114.3,
                            -122.0 if demo else -119.0, 0.01)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────
pred_col, info_col = st.columns([1, 1])

with pred_col:
    if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):
        input_data = {
            "MedInc": med_inc, "HouseAge": house_age,
            "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
            "Population": float(population), "AveOccup": ave_occup,
            "Latitude": latitude, "Longitude": longitude,
        }
        df_input = pd.DataFrame([input_data])[features]
        scaled   = scaler.transform(df_input)
        pred     = model.predict(scaled)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:1rem; opacity:0.8; margin-bottom:0.5rem;">
                Predicted Median House Value
            </div>
            <div class="pred-value">${pred * 100_000:,.0f}</div>
            <div class="pred-label">({pred:.4f} × $100,000 units)</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        st.markdown("<br>", unsafe_allow_html=True)
        pct = min(pred / 5.0, 1.0)
        color = "#2ecc71" if pred < 2 else "#f39c12" if pred < 4 else "#e74c3c"
        st.markdown(f"""
        <div style="background:#e8eaed; border-radius:999px; height:12px; margin-top:8px;">
            <div style="background:{color}; width:{pct*100:.1f}%; height:12px;
                        border-radius:999px; transition:width 0.5s;"></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:0.75rem; color:#5f6368; margin-top:4px;">
            <span>$0</span><span>$250k</span><span>$500k</span>
        </div>
        """, unsafe_allow_html=True)

        # Feature contribution mini chart
        if model is not None:
            coef = model.coef_
            feat_vals = scaled[0]
            contributions = coef * feat_vals

            fig, ax = plt.subplots(figsize=(5, 3.2))
            colors_bar = ["tomato" if c < 0 else "steelblue" for c in contributions]
            ax.barh(features, contributions, color=colors_bar)
            ax.axvline(0, color='black', linewidth=0.7)
            ax.set_title("Feature Contributions to This Prediction", fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()

        st.session_state["demo"] = False

with info_col:
    st.markdown("### 📐 How the Model Works")
    st.markdown("""
    The model uses **Ordinary Least Squares (OLS) Linear Regression**:

    ```
    Price = β₀ + β₁·MedInc + β₂·HouseAge + ... + β₈·Longitude
    ```

    Before prediction, features are **standardized** (mean=0, std=1)
    using a `StandardScaler` fitted on the training data.

    **Top predictors (standardized coefficients):**
    """)
    coef_data = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)
    coef_data["Impact"] = coef_data["Coefficient"].apply(
        lambda x: "🔵 Positive" if x > 0.05 else ("🔴 Negative" if x < -0.05 else "⚪ Neutral")
    )
    coef_data["Coefficient"] = coef_data["Coefficient"].round(4)
    st.dataframe(coef_data, hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9aa0a6; font-size:0.8rem;">
    Built by <b>Pranav Lakhe</b> · Maincrafts Technology AI/ML Internship · Task 1
    &nbsp;|&nbsp; Dataset: California Housing (sklearn) &nbsp;|&nbsp;
    <a href="https://github.com/Pranavv28" style="color:#1a73e8;">GitHub</a>
</div>
""", unsafe_allow_html=True)
