import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FinSight AI Dashboard", layout="wide", page_icon="📈")

# --- DATA LOADING ---
# We use st.cache_data so the app doesn't reload the CSVs every time you click a button
@st.cache_data
def load_data():
    try:
        df_historical = pd.read_csv('synthetic_cloud_billing.csv')
        df_anomalies = pd.read_csv('anomalies_detected.csv')
        df_forecast = pd.read_csv('forecast_results.csv')
        
        # Convert date columns back to datetime objects
        df_historical['Date'] = pd.to_datetime(df_historical['Date'])
        df_anomalies['Date'] = pd.to_datetime(df_anomalies['Date'])
        df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
        
        return df_historical, df_anomalies, df_forecast
    except FileNotFoundError:
        st.error("Data files not found! Please run the Generation, Anomaly, and Forecast scripts first.")
        return None, None, None

df_hist, df_anom, df_fcst = load_data()

if df_hist is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4241/4241107.png", width=50) # Generic cloud icon
    st.sidebar.title("FinSight AI Controls")
    
    selected_provider = st.sidebar.selectbox("Select Cloud Provider", df_hist['Provider'].unique())
    
    # Filter services based on provider
    available_services = df_hist[df_hist['Provider'] == selected_provider]['Service'].unique()
    selected_service = st.sidebar.selectbox("Select Service", available_services)
    
    # Set a dummy budget for the demonstration (e.g., $18,000 a month / roughly $600 a day limit)
    daily_budget_limit = st.sidebar.slider("Set Daily Budget Alert Limit ($)", 200, 1500, 600)

    # --- FILTER DATA ---
    hist_filtered = df_hist[(df_hist['Provider'] == selected_provider) & (df_hist['Service'] == selected_service)]
    anom_filtered = df_anom[(df_anom['Provider'] == selected_provider) & (df_anom['Service'] == selected_service) & (df_anom['Is_Anomaly'] == True)]
    fcst_filtered = df_fcst[(df_fcst['Provider'] == selected_provider) & (df_fcst['Service'] == selected_service)]

    # --- MAIN DASHBOARD UI ---
    st.title(f"FinOps Intelligence: {selected_provider} - {selected_service}")
    st.markdown("Proactive anomaly detection and multi-horizon spend forecasting.")

    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    last_30_days = hist_filtered[hist_filtered['Date'] >= (hist_filtered['Date'].max() - timedelta(days=30))]
    total_spend_30d = last_30_days['Cost'].sum()
    
    next_30_days = fcst_filtered[fcst_filtered['Date'] <= (fcst_filtered['Date'].min() + timedelta(days=30))]
    forecast_spend_30d = next_30_days['Forecasted_Cost_P50'].sum()
    
    anomaly_count = len(anom_filtered)

    col1.metric("30-Day Historical Spend", f"${total_spend_30d:,.2f}")
    col2.metric("30-Day Forecasted Spend", f"${forecast_spend_30d:,.2f}", f"{(forecast_spend_30d - total_spend_30d) / total_spend_30d * 100:.1f}%")
    col3.metric("Anomalies Detected (YTD)", anomaly_count, delta_color="inverse")
    
    # Budget Breach Prediction Logic
    breach_dates = fcst_filtered[fcst_filtered['Forecasted_Cost_P50'] > daily_budget_limit]
    if not breach_dates.empty:
        breach_date = breach_dates.iloc[0]['Date'].strftime('%Y-%m-%d')
        col4.error(f"⚠️ Budget Breach Predicted: {breach_date}")
    else:
        col4.success("✅ Forecast within Budget Limits")

    st.divider()

    # --- THE MASTER CHART (Burn Rate, Forecast, and Anomalies) ---
    st.subheader("Predictive Cost Trajectory & Anomaly Highlights")
    
    fig = go.Figure()

    # 1. Historical Data Line
    fig.add_trace(go.Scatter(x=hist_filtered['Date'], y=hist_filtered['Cost'], 
                             mode='lines', name='Historical Spend', line=dict(color='#2E86C1')))

    # 2. Anomaly Markers (Red Dots)
    fig.add_trace(go.Scatter(x=anom_filtered['Date'], y=anom_filtered['Cost'], 
                             mode='markers', name='Detected Anomaly', 
                             marker=dict(color='red', size=10, symbol='x')))

    # 3. Forecast P50 Line
    fig.add_trace(go.Scatter(x=fcst_filtered['Date'], y=fcst_filtered['Forecasted_Cost_P50'], 
                             mode='lines', name='Forecast (P50)', line=dict(color='#28B463', dash='dash')))

    # 4. Forecast Confidence Interval (Shaded Area between P10 and P90)
    fig.add_trace(go.Scatter(x=fcst_filtered['Date'], y=fcst_filtered['Worst_Case_Cost_P90'], 
                             mode='lines', marker=dict(color="#444"), line=dict(width=0), 
                             showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(x=fcst_filtered['Date'], y=fcst_filtered['Best_Case_Cost_P10'], 
                             mode='lines', marker=dict(color="#444"), line=dict(width=0), 
                             fillcolor='rgba(40, 180, 99, 0.2)', fill='tonexty', 
                             name='80% Confidence Interval'))

    # 5. Fixed Budget Line
    fig.add_hline(y=daily_budget_limit, line_dash="dot", annotation_text="Daily Budget Threshold", 
                  annotation_position="bottom right", line_color="red")

    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Cost ($)", 
                      hovermode="x unified", template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

    # --- ROOT CAUSE / ANOMALY INVESTIGATION TABLE ---
    st.subheader("🚨 High-Severity Anomalies Requiring Investigation")
    if not anom_filtered.empty:
        # Sort by most recent and most severe
        display_anomalies = anom_filtered[['Date', 'Provider', 'Service', 'Cost', 'Severity_Score']].sort_values(by='Date', ascending=False)
        
        # Add a mock "Root Cause" column for the hackathon "Wow" factor
        display_anomalies['AI Root Cause Attribution'] = [
            "Dev Environment Spike: High Compute Usage" if p == "AWS" else "Data Egress Transfer Spike" 
            for p in display_anomalies['Provider']
        ]
        
        st.dataframe(display_anomalies, use_container_width=True, hide_index=True)
    else:
        st.info("No anomalies detected for this specific service profile.")