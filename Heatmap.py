import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

# Streamlit App
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="centered")
st.title("Black-Scholes Option Pricing Dashboard")

st.sidebar.header("Input Parameters")

premium = st.sidebar.slider("Premium", 0.0, 5.0, 2.5, 0.01)
S = st.sidebar.slider("Underlying Price (S)", 0, 1000, 50, 1)
K = st.sidebar.slider("Strike Price (K)", 0, 1000, 55, 1)
T = st.sidebar.slider("Time to Expiration (T in years)", 0.01, 10.0, 1.0, 0.01)
r_percent = st.sidebar.slider("Risk-Free Rate (r) [%]", 0.0, 100.0, 2.0, 0.01)
r = r_percent / 100

# New Heatmap Controls
st.subheader("Heatmap Parameters")
col1, col2 = st.columns(2)
with col1:
    k_min, k_max = st.slider("Strike Price Range (K)", 0, 200, (40, 70), step=1)
    k_step = st.number_input("Strike Price Step", 1, 50, 5)
with col2:
    v_min, v_max = st.slider("Volatility Range (%)", 5, 200, (10, 100), step=5)
    v_step = st.number_input("Volatility Step (%)", 1, 50, 10)

if st.button("Calculate Option Prices & Heatmap"):
    st.subheader("Inputs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Premium", f"{premium}")
    col2.metric("Underlying Price (S)", f"{S}")
    col3.metric("Strike Price (K)", f"{K}")

    col4, col5 = st.columns(2)
    col4.metric("Time to Expiry (T)", f"{T} years")
    col5.metric("Risk-Free Rate (r)", f"{r:.2%}")

    # Build heatmap matrix
    k_values = np.arange(k_min, k_max + k_step, k_step)
    vol_values = np.arange(v_min / 100, v_max / 100 + v_step / 100, v_step / 100)

    heatmap_data = []
    for vol in vol_values:
        row = []
        for k in k_values:
            call_pnl, _ = calculate_option_prices(S, k, T, r, vol, premium)
            row.append(call_pnl)
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=[f"{v*100:.0f}%" for v in vol_values], columns=[f"{k}" for k in k_values])

    st.subheader("Call Option P&L Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={'label': 'Call P&L'}
    )
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Volatility")
    st.pyplot(fig)
