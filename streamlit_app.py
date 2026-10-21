# importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# setting up page configuration
st.set_page_config(
    page_title="Digital Twin Engine Simulator",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("## Methanol/Natural Gas Dual-Fuel Engine Digital Twin 🔥")
st.markdown("A real-time simulation tool predicting performance and emissions in real-time, powered by machine learning models trained on the WP10NG336E40 Spark-Ignition NG Engine.")

# loading the ML models and caching them
@st.cache_resource
def load_models():
    """Loads the trained machine learning models into server RAM once."""
    return {
            'BSNOX': joblib.load('trained_models/model_BSNOX.pkl'),
            'BTE': joblib.load('trained_models/model_BTE.pkl'),
            'BSTHC': joblib.load('trained_models/model_BSTHC.pkl'),
            'BSCO2': joblib.load('trained_models/model_BSCO2.pkl')
        }

models = load_models()


# initializing session state for the natural gas & methanol sliders
if 'ng_weight' not in st.session_state:
    st.session_state.ng_weight = 100
if 'meth_weight' not in st.session_state:
    st.session_state.meth_weight = 0

def update_from_ng():
    st.session_state.meth_weight = 100 - st.session_state.ng_weight

def update_from_meth():
    st.session_state.ng_weight = 100 - st.session_state.meth_weight


st.divider()

with st.expander("🎛️ Tune the Engine Load & Fuel Blend", expanded=True):
    # implementing 2D Interpolation for the MSR
    known_loads = [0.387, 0.775, 1.163]
    known_mass_ratios = [0, 20, 40, 60, 80]

    # defining the physical MSR grid (Rows = Loads, Columns = Mass Ratios)
    msr_grid = np.array([
        [0.0, 7.7, 14.2, 20.1, 24.4], # 0.387 MPa row
        [0.0, 7.5, 14.0, 19.4, 23.9], # 0.775 MPa row
        [0.0, 7.2, 13.8, 19.1, 24.2]  # 1.163 MPa row
    ])

    # Create 3 columns for the layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_bmep = st.slider(
            "Mechanical Load (BMEP in MPa)", 
            min_value=0.387, 
            max_value=1.163, 
            value=0.78, 
            step=0.01
        )

    with col2:
        ng_slider = st.slider(
            "Natural Gas (NG) %", 
            min_value=20, 
            max_value=100, 
            step=1, 
            key='ng_weight', 
            on_change=update_from_ng
        )

    with col3:
        meth_slider = st.slider(
            "Methanol %", 
            min_value=0, 
            max_value=80, 
            step=1, 
            key='meth_weight', 
            on_change=update_from_meth
        )

    # interpolating across the BMEP axis to generate a custom MSR array for the input load
    custom_load_msr_array = [
        np.interp(selected_bmep, known_loads, msr_grid[:, i]) 
        for i in range(len(known_mass_ratios))
    ]

    # interpolating across the Methanol mass percentage using the newly generated custom array
    actual_msr = np.interp(st.session_state.meth_weight, known_mass_ratios, custom_load_msr_array)


    # Display the resulting MSR directly below the sliders before the graphs start
    st.metric(label="Effective Methanol Energy (MSR)", value=f"{actual_msr:.2f}%")

st.divider() # Creates a visual break before your Tier 1, 2, and 3 graphs begin



# making baseline prediction (0% Methanol) to calculate deltas
X_baseline = np.array([[selected_bmep, 0.0]])
base_bte = models['BTE'].predict(X_baseline)[0]
base_nox = models['BSNOX'].predict(X_baseline)[0]
base_thc = models['BSTHC'].predict(X_baseline)[0]
base_co2 = models['BSCO2'].predict(X_baseline)[0]

# making current prediction (based on input MSR setting)
X_current = np.array([[selected_bmep, actual_msr]])
curr_bte = models['BTE'].predict(X_current)[0]
curr_nox = models['BSNOX'].predict(X_current)[0]
curr_thc = models['BSTHC'].predict(X_current)[0]
curr_co2 = models['BSCO2'].predict(X_current)[0]


# setting section for predicted engine outputs i.e. performance and emissions
with st.expander("📊 Real-Time Engine Output", expanded=False):
    # Changing from 4 columns to 2 columns for a clean 2x2 mobile grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Brake Thermal Efficiency (BTE)", f"{curr_bte:.2f} %", f"{curr_bte - base_bte:.2f} %")
        st.metric("BSTHC Emissions", f"{curr_thc:.2f} g/kWh", f"{curr_thc - base_thc:.2f} g/kWh", delta_color="inverse")
        
    with col2:
        st.metric("BSNOx Emissions", f"{curr_nox:.2f} g/kWh", f"{curr_nox - base_nox:.2f} g/kWh", delta_color="inverse")
        st.metric("BSCO2 Emissions", f"{curr_co2:.1f} g/kWh", f"{curr_co2 - base_co2:.1f} g/kWh", delta_color="inverse")

st.divider()

# setting plotly graph for interactive display of engine output and optimal operating boundaries
st.subheader("🧭Interactive Engine Tuning Map")

# generating the Local Search Space for the current BMEP
msr_sweep = np.linspace(0, 25, 300)
X_local = np.c_[np.full_like(msr_sweep, selected_bmep), msr_sweep]

local_nox = models['BSNOX'].predict(X_local)
local_bte = models['BTE'].predict(X_local)

# extracting the non-dominated points
df_local = pd.DataFrame({'BSNOX': local_nox, 'BTE': local_bte, 'MSR': msr_sweep})
df_local = df_local.sort_values('BSNOX', ascending=True)

front_x, front_y, front_msr = [], [], []
max_bte_seen = -np.inf

for _, row in df_local.iterrows():
    if row['BTE'] > max_bte_seen:
        front_x.append(row['BSNOX'])
        front_y.append(row['BTE'])
        front_msr.append(row['MSR'])
        max_bte_seen = row['BTE']

# building the Plotly figure
fig_pareto = go.Figure()

# plotting the full background search space
fig_pareto.add_trace(go.Scatter(
    x=local_nox, y=local_bte,
    mode='markers',
    marker=dict(size=4, color='lightgray'),
    name='Continuous Search Space',
    hoverinfo='skip'
))

# plotting the optimal Pareto line
fig_pareto.add_trace(go.Scatter(
    x=front_x, y=front_y,
    mode='lines+markers',
    line=dict(color='red', width=3, dash='dash'),
    marker=dict(symbol='star', size=8, color='red'),
    name='Optimal Tuning Boundary',
    hovertemplate="NOx: %{x:.2f}<br>BTE: %{y:.2f}%<br>Optimal MSR: %{text:.1f}%<extra></extra>",
    text=front_msr
))

# overlaying the current tuning marker based on input i.e. engine load and MSR
fig_pareto.add_trace(go.Scatter(
    x=[curr_nox], y=[curr_bte],
    mode='markers',
    marker=dict(symbol='circle-cross', size=20, color='gold', line=dict(color='black', width=2)),
    name=f'Current Tuning ({actual_msr:.1f}% MSR)',
    hovertemplate="Current NOx: %{x:.2f}<br>Current BTE: %{y:.2f}%<extra></extra>"
))

fig_pareto.update_layout(
    xaxis=dict(title='Predicted BSNOX [g/(kW·h)] - (Lower is Better)', fixedrange=True),
    yaxis=dict(title='Predicted BTE [%] - (Higher is Better)', fixedrange=True),
    height=500,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode='closest',
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
)

# Render the Pareto chart with the toolbar disabled
st.plotly_chart(fig_pareto, use_container_width=True, config={'displayModeBar': True})

st.divider()


# setting the response curve sections
st.subheader("📈Performance & Emission Curves")
col_bte, col_nox = st.columns(2)

# generating the BTE response curve
fig_bte = go.Figure()
fig_bte.add_trace(go.Scatter(x=msr_sweep, y=local_bte, mode='lines', line=dict(color='blue', width=3)))
fig_bte.add_vline(x=actual_msr, line_dash="dash", line_color="gold", annotation_text="Current Tuning")
fig_bte.update_layout(
    title="Thermal Efficiency Response", 
    xaxis=dict(title="MSR (%)", fixedrange=True), 
    yaxis=dict(title="BTE (%)", fixedrange=True), 
    height=350, 
    margin=dict(l=0, r=0, t=40, b=0)
)

# generating the NOx response curve
fig_nox = go.Figure()
fig_nox.add_trace(go.Scatter(x=msr_sweep, y=local_nox, mode='lines', line=dict(color='red', width=3)))
fig_nox.add_vline(x=actual_msr, line_dash="dash", line_color="gold", annotation_text="Current Tuning")
fig_nox.update_layout(
    title="Thermal NOx Formation", 
    xaxis=dict(title="MSR (%)", fixedrange=True), 
    yaxis=dict(title="BSNOx (g/kWh)", fixedrange=True), 
    height=350, 
    margin=dict(l=0, r=0, t=40, b=0)
)

# Render the response curves with the toolbars disabled
with col_bte:
    st.plotly_chart(fig_bte, use_container_width=True, config={'displayModeBar': True})
with col_nox:
    st.plotly_chart(fig_nox, use_container_width=True, config={'displayModeBar': True})