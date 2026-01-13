# logic.py
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime
import logging
from typing import Tuple, Dict
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration class for easy threshold management
class AnomalyConfig:
    """Configuration for anomaly detection thresholds"""
    RATIO_THRESHOLD = 2.0
    RATIO_SCORE_WEIGHT = 30
    
    BIO_UPDATE_THRESHOLD = 50
    GHOST_PATTERN_WEIGHT = 40
    
    DEVIATION_MULTIPLIER = 30
    DEVIATION_WEIGHT = 30
    
    SPIKE_STD_MULTIPLIER = 3
    ROLLING_WINDOW = 7
    
    MAX_RISK_SCORE = 100

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize and validate input dataframe
    Prevents injection attacks and data corruption
    """
    if df.empty:
        logger.warning("Empty dataframe provided")
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names (map from CSV format to internal format)
    column_mapping = {
        'Adult_Enrolment': 'enrol_adult',
        'Child_Enrolment': 'enrol_child',
        'Bio_Update': 'bio_update',
        'Demo_Update': 'demo_update',
        'Date': 'date',
        'State': 'state',
        'Pincode': 'pincode',
        'OperatorID': 'operator_id'
    }
    
    # Apply mapping if columns exist
    df = df.rename(columns=column_mapping)
    
    # Validate required columns exist
    required_cols = ['enrol_adult', 'enrol_child', 'bio_update', 'demo_update', 
                     'state', 'pincode', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert numeric columns and handle errors
    numeric_cols = ['enrol_adult', 'enrol_child', 'bio_update', 'demo_update', 'pincode']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace NaN with 0 for numeric columns (after logging)
    nan_counts = df[numeric_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found and replaced with 0: {nan_counts[nan_counts > 0].to_dict()}")
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Ensure non-negative values
    for col in numeric_cols:
        if (df[col] < 0).any():
            logger.warning(f"Negative values found in {col}, setting to 0")
            df[col] = df[col].clip(lower=0)
    
    # Sanitize string columns (prevent injection)
    string_cols = ['state', 'operator_id']
    for col in string_cols:
        if col in df.columns:
            # Remove special characters that could be used for injection
            df[col] = df[col].astype(str).str.replace(r'[<>\"\';&|`$]', '', regex=True)
            # Limit length to prevent buffer overflow
            df[col] = df[col].str[:100]
    
    # Validate date format
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"{invalid_dates} invalid dates found and removed")
            df = df[df['date'].notna()]
    except Exception as e:
        logger.error(f"Date conversion failed: {e}")
        raise ValueError(f"Date column format invalid: {e}")
    
    logger.info(f"Sanitized dataframe: {len(df)} records, {len(df.columns)} columns")
    return df

def calculate_statistical_baseline(df: pd.DataFrame, group_col: str, 
                                   metric_cols: list) -> pd.DataFrame:
    """
    Calculate statistical baselines with outlier removal
    Uses IQR method to create robust baselines
    """
    baselines = {}
    
    for metric in metric_cols:
        # Group statistics
        grouped = df.groupby(group_col)[metric]
        
        # Calculate Q1, Q3, and IQR for outlier removal
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds (1.5 * IQR is standard outlier detection)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Calculate mean excluding outliers
        def robust_mean(series):
            lb = lower_bound[series.name]
            ub = upper_bound[series.name]
            filtered = series[(series >= lb) & (series <= ub)]
            return filtered.mean() if len(filtered) > 0 else series.mean()
        
        baselines[f'{group_col}_avg_{metric}'] = grouped.transform(robust_mean)
        baselines[f'{group_col}_std_{metric}'] = grouped.transform('std')
    
    return pd.DataFrame(baselines)

def detect_anomalies_pro(df: pd.DataFrame, config: AnomalyConfig = None) -> pd.DataFrame:
    """
    Advanced anomaly detection with multiple algorithms
    
    Features:
    - Statistical baseline comparison (robust to outliers)
    - Multi-factor risk scoring
    - Ghost pattern detection
    - Deviation analysis with z-scores
    
    Args:
        df: Input dataframe with enrollment data
        config: Configuration object with detection thresholds
    
    Returns:
        DataFrame with anomaly scores and flags
    """
    if config is None:
        config = AnomalyConfig()
    
    try:
        # Sanitize input
        df = sanitize_dataframe(df)
        
        if df.empty:
            logger.warning("Empty dataframe after sanitization")
            return df
        
        logger.info(f"Starting anomaly detection on {len(df)} records")
        
        # 1. Calculate State Baselines (Robust)
        baseline_metrics = calculate_statistical_baseline(
            df, 'state', ['enrol_adult', 'enrol_child', 'bio_update']
        )
        df = pd.concat([df, baseline_metrics], axis=1)
        
        # 2. Adult-to-Child Ratio Analysis (with safety checks)
        # Add 1 to denominator to avoid division by zero
        df['ratio_score'] = df['enrol_adult'] / (df['enrol_child'] + 1)
        df['ratio_score'] = df['ratio_score'].replace([np.inf, -np.inf], 0)
        
        # 3. Deviation from State Baseline (Z-score method)
        # More statistically rigorous than simple multiplication
        df['deviation_score'] = np.where(
            df['state_std_enrol_adult'] > 0,
            (df['enrol_adult'] - df['state_avg_enrol_adult']) / df['state_std_enrol_adult'],
            0
        )
        df['deviation_score'] = df['deviation_score'].abs()
        
        # 4. Ghost Pattern Detection
        # High bio updates with zero demo updates suggests fake activity
        df['is_ghost_pattern'] = (
            (df['bio_update'] > config.BIO_UPDATE_THRESHOLD) & 
            (df['demo_update'] == 0)
        )
        
        # 5. Volume Anomaly Detection
        # Flag centers with unusually high total activity
        df['total_activity'] = df['enrol_adult'] + df['enrol_child'] + df['bio_update']
        activity_baseline = calculate_statistical_baseline(
            df, 'state', ['total_activity']
        )
        df['activity_z_score'] = np.where(
            activity_baseline['state_std_total_activity'] > 0,
            (df['total_activity'] - activity_baseline['state_avg_total_activity']) / 
            activity_baseline['state_std_total_activity'],
            0
        )
        
        # 6. Comprehensive Risk Scoring
        def calculate_risk_score(row) -> float:
            """Calculate composite risk score from multiple factors"""
            score = 0
            
            # Factor 1: Unusual ratio
            if row['ratio_score'] > config.RATIO_THRESHOLD:
                score += config.RATIO_SCORE_WEIGHT
            
            # Factor 2: Ghost pattern (weighted heavily)
            if row['is_ghost_pattern']:
                score += config.GHOST_PATTERN_WEIGHT
            
            # Factor 3: Baseline deviation (z-score > 2 is significant)
            if row['deviation_score'] > 2:
                score += config.DEVIATION_WEIGHT * min(row['deviation_score'] / 2, 1)
            
            # Factor 4: Activity anomaly
            if abs(row['activity_z_score']) > 2:
                score += 20
            
            # Factor 5: Suspicious zero values
            if row['enrol_adult'] > 100 and row['enrol_child'] == 0:
                score += 15
            
            return min(score, config.MAX_RISK_SCORE)
        
        df['risk_score'] = df.apply(calculate_risk_score, axis=1)
        
        # 7. Categorize risk levels
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Log summary statistics
        risk_summary = df['risk_level'].value_counts()
        logger.info(f"Risk distribution: {risk_summary.to_dict()}")
        logger.info(f"High-risk centers: {len(df[df['risk_score'] > 70])}")
        
        return df
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}", exc_info=True)
        raise

def detect_temporal_spikes(df: pd.DataFrame, config: AnomalyConfig = None) -> pd.DataFrame:
    """
    Identify sudden temporal spikes using statistical methods
    
    Features:
    - Rolling statistics with configurable window
    - Z-score based spike detection
    - Handles missing data gracefully
    
    Args:
        df: Input dataframe with time series data
        config: Configuration object
    
    Returns:
        DataFrame containing only spike records
    """
    if config is None:
        config = AnomalyConfig()
    
    try:
        df = sanitize_dataframe(df)
        
        if df.empty:
            logger.warning("Empty dataframe for spike detection")
            return df
        
        logger.info(f"Starting spike detection on {len(df)} records")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by pincode and date for rolling calculations
        df = df.sort_values(by=['pincode', 'date'])
        
        # Calculate rolling statistics per pincode
        # Using exponential moving average for better recent trend capture
        df['rolling_mean'] = (df.groupby('pincode')['bio_update']
                              .transform(lambda x: x.rolling(
                                  window=config.ROLLING_WINDOW, 
                                  min_periods=1
                              ).mean()))
        
        df['rolling_std'] = (df.groupby('pincode')['bio_update']
                            .transform(lambda x: x.rolling(
                                window=config.ROLLING_WINDOW, 
                                min_periods=1
                            ).std()))
        
        # Handle cases where std is 0 or NaN
        df['rolling_std'] = df['rolling_std'].fillna(1)
        df['rolling_std'] = df['rolling_std'].replace(0, 1)
        
        # Calculate Z-score for spike detection
        df['spike_z_score'] = (df['bio_update'] - df['rolling_mean']) / df['rolling_std']
        
        # Define spike: Z-score > threshold
        df['is_spike'] = df['spike_z_score'] > config.SPIKE_STD_MULTIPLIER
        
        # Additional condition: absolute value must be significant
        df['is_spike'] = df['is_spike'] & (df['bio_update'] > df['rolling_mean'] * 1.5)
        
        # Calculate spike severity
        df['spike_severity'] = np.where(
            df['is_spike'],
            ((df['bio_update'] - df['rolling_mean']) / df['rolling_mean'] * 100),
            0
        )
        
        # Filter only spikes
        spikes_df = df[df['is_spike'] == True].copy()
        
        # Sort by severity
        spikes_df = spikes_df.sort_values(by='spike_severity', ascending=False)
        
        logger.info(f"Detected {len(spikes_df)} temporal spikes")
        
        return spikes_df
        
    except Exception as e:
        logger.error(f"Spike detection failed: {e}", exc_info=True)
        raise

def generate_audit_pdf(anomalies_df: pd.DataFrame, spikes_df: pd.DataFrame) -> bytes:
    """
    Generate professional PDF audit report with enhanced formatting
    
    Features:
    - Executive summary with key metrics
    - Top risk centers with detailed analysis
    - Temporal spike analysis
    - Recommendations section
    - Proper error handling
    
    Args:
        anomalies_df: DataFrame with anomaly analysis results
        spikes_df: DataFrame with temporal spike data
    
    Returns:
        PDF as bytes
    """
    try:
        # Sanitize inputs
        anomalies_df = sanitize_dataframe(anomalies_df)
        spikes_df = sanitize_dataframe(spikes_df) if not spikes_df.empty else spikes_df
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(0, 15, txt="AADHAR OPERATIONS AUDIT REPORT", ln=True, align='C')
        
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ln=True, align='C')
        pdf.ln(5)
        
        # Divider line
        pdf.set_draw_color(0, 0, 128)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, txt="EXECUTIVE SUMMARY", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", '', 11)
        
        # Calculate metrics safely
        total_centers = len(anomalies_df)
        high_risk = len(anomalies_df[anomalies_df['risk_score'] > 70]) if 'risk_score' in anomalies_df.columns else 0
        medium_risk = len(anomalies_df[(anomalies_df['risk_score'] > 30) & 
                                       (anomalies_df['risk_score'] <= 70)]) if 'risk_score' in anomalies_df.columns else 0
        ghost_patterns = anomalies_df['is_ghost_pattern'].sum() if 'is_ghost_pattern' in anomalies_df.columns else 0
        total_spikes = len(spikes_df)
        
        summary_metrics = [
            f"Total Centers Analyzed: {total_centers}",
            f"High-Risk Centers (Score > 70): {high_risk}",
            f"Medium-Risk Centers (Score 30-70): {medium_risk}",
            f"Ghost Patterns Detected: {ghost_patterns}",
            f"Temporal Spikes Detected: {total_spikes}",
        ]
        
        for metric in summary_metrics:
            pdf.cell(0, 8, txt=f"  • {metric}", ln=True)
        
        pdf.ln(8)
        
        # Critical Risk Centers
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(178, 34, 34)
        pdf.cell(0, 10, txt="CRITICAL RISK CENTERS", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(0, 0, 0)
        
        if high_risk > 0:
            top_risks = anomalies_df.nlargest(10, 'risk_score')
            
            for idx, (_, row) in enumerate(top_risks.iterrows(), 1):
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(0, 7, txt=f"{idx}. Operator: {row.get('operator_id', 'N/A')} | " +
                                  f"Location: {row.get('state', 'N/A')}-{row.get('pincode', 'N/A')}", 
                        ln=True)
                
                pdf.set_font("Arial", '', 9)
                pdf.set_text_color(50, 50, 50)
                
                risk_details = f"    Risk Score: {row['risk_score']:.0f}/100 | " + \
                              f"Deviation: {row.get('deviation_score', 0):.1f}σ | " + \
                              f"Ratio: {row.get('ratio_score', 0):.2f}"
                
                pdf.cell(0, 6, txt=risk_details, ln=True)
                
                # Add issue description
                issues = []
                if row.get('is_ghost_pattern', False):
                    issues.append("Ghost Pattern")
                if row.get('deviation_score', 0) > 2:
                    issues.append("High Baseline Deviation")
                if row.get('ratio_score', 0) > 2:
                    issues.append("Unusual Adult/Child Ratio")
                
                if issues:
                    pdf.set_text_color(178, 34, 34)
                    pdf.cell(0, 6, txt=f"    Issues: {', '.join(issues)}", ln=True)
                
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
        else:
            pdf.cell(0, 8, txt="  ✓ No critical risk centers detected.", ln=True)
        
        pdf.ln(5)
        
        # Temporal Spikes Section
        if total_spikes > 0:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(255, 140, 0)
            pdf.cell(0, 10, txt="TEMPORAL SPIKE ANALYSIS", ln=True)
            pdf.ln(3)
            
            pdf.set_font("Arial", '', 10)
            pdf.set_text_color(0, 0, 0)
            
            top_spikes = spikes_df.nlargest(10, 'spike_severity') if 'spike_severity' in spikes_df.columns else spikes_df.head(10)
            
            for idx, (_, row) in enumerate(top_spikes.iterrows(), 1):
                spike_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
                pdf.cell(0, 7, txt=f"{idx}. Date: {spike_date} | " +
                                  f"Pincode: {row.get('pincode', 'N/A')} | " +
                                  f"State: {row.get('state', 'N/A')}", ln=True)
                
                pdf.set_font("Arial", '', 9)
                pdf.set_text_color(50, 50, 50)
                
                spike_info = f"    Bio Updates: {row.get('bio_update', 0):.0f} | " + \
                            f"Rolling Avg: {row.get('rolling_mean', 0):.1f} | " + \
                            f"Severity: {row.get('spike_severity', 0):.1f}%"
                
                pdf.cell(0, 6, txt=spike_info, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
        
        # Recommendations
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, txt="RECOMMENDATIONS", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0)
        
        recommendations = [
            "1. Immediate Investigation: All centers with risk scores > 70 require immediate audit.",
            "2. Ghost Pattern Analysis: Centers showing high bio updates with zero demo updates need verification.",
            "3. Baseline Monitoring: Establish automated alerts for centers exceeding 2σ from state baseline.",
            "4. Temporal Monitoring: Implement real-time spike detection for early fraud prevention.",
            "5. Operator Training: High-risk operators should undergo compliance retraining.",
        ]
        
        for rec in recommendations:
            pdf.multi_cell(0, 7, txt=rec)
            pdf.ln(2)
        
        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, txt="This is an automated report generated by Aadhar Sentinel Pro", 
                ln=True, align='C')
        pdf.cell(0, 5, txt="For questions, contact: audit@uidai.gov.in", 
                ln=True, align='C')
        
        # Generate PDF bytes with proper encoding
        pdf_output = pdf.output(dest='S')
        
        # Convert to bytes properly
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = bytes(pdf_output)
        
        logger.info(f"PDF report generated successfully: {len(pdf_bytes)} bytes")
        return pdf_bytes
     
