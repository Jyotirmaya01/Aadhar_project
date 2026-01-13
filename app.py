# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import database
import logic
import uuid
from datetime import datetime

# --- CONFIG ---
st.set_page_config(
    page_title="Aadhar Sentinel Pro", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Initialize database
try:
    database.init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")
    st.stop()

# --- ENHANCED CSS (Modern Dark Theme with Glassmorphism) ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp { 
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Improved Cards with Glassmorphism */
    .card { 
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 200, 255, 0.2);
    }
    
    /* Enhanced Metrics */
    .metric-val { 
        font-size: 32px; 
        font-weight: 700; 
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .metric-val.danger {
        background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-val.warning {
        background: linear-gradient(135deg, #FFB74D 0%, #FFA726 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-lbl { 
        font-size: 13px; 
        color: #9CA3AF; 
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Enhanced Buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none;
        font-weight: 600;
        padding: 12px 32px;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(0, 201, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(0, 201, 255, 0.5);
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #00C9FF;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 30, 46, 0.5);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(0, 201, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("# 🛡️ Sentinel Pro")
st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigation Module", 
    ["📂 Bulk Upload", "📊 Analytics Dashboard", "📑 Audit Reports"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success("🟢 Database: Connected")
st.sidebar.info(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}")

# --- HELPER FUNCTION: Validate CSV ---
def validate_csv(df):
    """Validate uploaded CSV structure and data quality"""
    required_cols = ['OperatorID', 'Date', 'State', 'Pincode', 
                     'Adult_Enrolment', 'Child_Enrolment', 'Bio_Update', 'Demo_Update']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    
    # Check for nulls in critical columns
    critical_cols = ['OperatorID', 'Date', 'State']
    for col in critical_cols:
        if df[col].isnull().any():
            return False, f"Column '{col}' contains null values"
    
    # Validate date format
    try:
        pd.to_datetime(df['Date'])
    except:
        return False, "Date column format invalid. Use YYYY-MM-DD"
    
    return True, "Validation passed"

# --- 1. BULK UPLOAD MODULE ---
if nav == "📂 Bulk Upload":
    st.title("📂 Secure Data Ingestion")
    st.markdown("Upload daily CSV reports with automatic schema validation and integrity checks.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Template Download
        template_data = pd.DataFrame(columns=[
            'OperatorID', 'Date', 'State', 'Pincode', 
            'Adult_Enrolment', 'Child_Enrolment', 'Bio_Update', 'Demo_Update'
        ])
        
        # Add sample row for clarity
        template_data.loc[0] = ['OP001', '2024-01-15', 'Maharashtra', '400001', 150, 45, 20, 10]
        
        st.download_button(
            "📥 Download CSV Template", 
            template_data.to_csv(index=False), 
            "aadhar_template.csv", 
            "text/csv",
            help="Download the standard template with correct column names"
        )
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>📋 Required Fields</h4>
            <ul style='font-size: 12px; color: #AAA;'>
                <li>OperatorID</li>
                <li>Date (YYYY-MM-DD)</li>
                <li>State, Pincode</li>
                <li>Enrolment & Update counts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Daily Report", 
        type=['csv'],
        help="Drag and drop or click to select CSV file"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validation
            is_valid, msg = validate_csv(df)
            
            if not is_valid:
                st.error(f"❌ Validation Failed: {msg}")
            else:
                st.success(f"✅ {msg}")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Total Records", len(df))
                col_stat2.metric("Unique Operators", df['OperatorID'].nunique())
                col_stat3.metric("Date Range", f"{df['Date'].min()} to {df['Date'].max()}")
                
                # Process button
                if st.button("🚀 Process & Ingest Data", use_container_width=True):
                    with st.spinner("Processing data... Please wait."):
                        try:
                            batch_id = str(uuid.uuid4())
                            database.insert_batch_data(df, batch_id)
                            
                            st.success(f"✅ Successfully ingested {len(df)} records!")
                            st.info(f"📦 Batch ID: `{batch_id[:8]}...`")
                            
                            # Show success metrics
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Database insertion failed: {e}")
                
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty.")
        except Exception as e:
            st.error(f"❌ Error parsing file: {e}")

# --- 2. ANALYTICS DASHBOARD ---
elif nav == "📊 Analytics Dashboard":
    st.title("📊 Forensic Analytics Suite")
    st.markdown("Real-time anomaly detection and pattern analysis powered by ML algorithms")
    
    # Fetch data with error handling
    try:
        df = database.fetch_time_series_data()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()
    
    if df.empty:
        st.warning("⚠️ No data found. Please upload data in the Bulk Upload module first.")
        st.stop()
    
    # Run Analysis with progress indicator
    with st.spinner("🔍 Running forensic analysis..."):
        try:
            analyzed_df = logic.detect_anomalies_pro(df)
            spikes_df = logic.detect_temporal_spikes(df)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()
    
    # --- Top KPIs with Enhanced Design ---
    st.subheader("🎯 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    
    total_records = len(df)
    high_risk_count = len(analyzed_df[analyzed_df['risk_score'] > 70]) if 'risk_score' in analyzed_df.columns else 0
    ghost_patterns = len(analyzed_df[analyzed_df['bio_update'] > 50]) if 'bio_update' in analyzed_df.columns else 0
    spike_count = len(spikes_df)
    
    c1.markdown(f"""
    <div class='card'>
        <div class='metric-lbl'>Total Records</div>
        <div class='metric-val'>{total_records:,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class='card'>
        <div class='metric-lbl'>High Risk Centers</div>
        <div class='metric-val danger'>{high_risk_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
    <div class='card'>
        <div class='metric-lbl'>Ghost Patterns</div>
        <div class='metric-val warning'>{ghost_patterns}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class='card'>
        <div class='metric-lbl'>Temporal Spikes</div>
        <div class='metric-val warning'>{spike_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Enhanced Visuals ---
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("📈 Trend Analysis: Biometric Anomalies Over Time")
        
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                daily_vol = df.groupby('date')[['bio_update', 'enrol_adult']].sum().reset_index()
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=daily_vol['date'], 
                    y=daily_vol['bio_update'],
                    mode='lines+markers',
                    name='Bio Updates',
                    line=dict(color='#00C9FF', width=3),
                    marker=dict(size=6)
                ))
                fig_trend.add_trace(go.Scatter(
                    x=daily_vol['date'], 
                    y=daily_vol['enrol_adult'],
                    mode='lines+markers',
                    name='Adult Enrolments',
                    line=dict(color='#92FE9D', width=3),
                    marker=dict(size=6)
                ))
                
                fig_trend.update_layout(
                    template="plotly_dark",
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating trend chart: {e}")
        else:
            st.warning("Date column not found in data")
    
    with col_side:
        st.subheader("📍 State Deviation Index")
        
        if 'state' in analyzed_df.columns and 'deviation_score' in analyzed_df.columns:
            state_dev = (analyzed_df.groupby('state')['deviation_score']
                        .mean()
                        .reset_index()
                        .sort_values(by='deviation_score', ascending=False)
                        .head(8))
            
            fig_bar = px.bar(
                state_dev, 
                x='state', 
                y='deviation_score',
                color='deviation_score',
                color_continuous_scale='Reds',
                template="plotly_dark"
            )
            
            fig_bar.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Required columns not found for state analysis")

    # --- Detailed Spike View ---
    st.markdown("---")
    st.subheader("⚡ Detected Temporal Spikes (Sudden Surges)")
    
    if not spikes_df.empty:
        # Filter to show only actual spikes
        display_cols = [col for col in ['date', 'pincode', 'state', 'bio_update', 'rolling_mean', 'is_spike'] 
                       if col in spikes_df.columns]
        
        if display_cols:
            spike_display = spikes_df[display_cols]
            if 'rolling_mean' in spike_display.columns:
                spike_display = spike_display.style.format({'rolling_mean': '{:.2f}'})
            
            st.dataframe(spike_display, use_container_width=True)
        else:
            st.info("Spike data columns not found")
    else:
        st.success("✅ No significant temporal spikes detected")

# --- 3. AUDIT REPORTS ---
elif nav == "📑 Audit Reports":
    st.title("📑 Automated Audit Generation")
    st.markdown("Generate comprehensive compliance reports with forensic analysis results")
    
    try:
        df = database.fetch_time_series_data()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()
    
    if df.empty:
        st.error("❌ No data available. Please upload data first in the Bulk Upload module.")
        st.stop()
    
    st.info(f"📊 Generating report from {len(df):,} records...")
    
    with st.spinner("🔄 Analyzing data and generating PDF..."):
        try:
            analyzed_df = logic.detect_anomalies_pro(df)
            spikes_df = logic.detect_temporal_spikes(df)
            
            # Generate PDF
            pdf_bytes = logic.generate_audit_pdf(analyzed_df, spikes_df)
            
            st.success("✅ Report Generated Successfully!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class='card'>
                    <h4>📄 Report Details</h4>
                    <ul style='font-size: 14px; color: #CCC;'>
                        <li>Total Records Analyzed: <b>{}</b></li>
                        <li>Anomalies Detected: <b>{}</b></li>
                        <li>Temporal Spikes: <b>{}</b></li>
                        <li>Report Generated: <b>{}</b></li>
                    </ul>
                </div>
                """.format(
                    len(analyzed_df),
                    len(analyzed_df[analyzed_df['risk_score'] > 70]) if 'risk_score' in analyzed_df.columns else 0,
                    len(spikes_df),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ), unsafe_allow_html=True)
            
            with col2:
                st.download_button(
                    label="📥 Download Official Audit PDF",
                    data=pdf_bytes,
                    file_name=f"aadhar_audit_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Report generation failed: {e}")
            st.info("Please check your database connection and data integrity.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛡️ Aadhar Sentinel Pro v2.0 | Powered by Advanced ML Analytics</p>
</div>
""", unsafe_allow_html=True)
