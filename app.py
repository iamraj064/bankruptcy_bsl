import sqlite3
import logging
import sys
import json
import re
import hashlib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from config import call_llm, call_llm_haiku, call_llm_with_cache, count_token_usage
from dotenv import load_dotenv
from insights_generator import generate_insights
from forecasting_engine import (
    load_monthly_series,
    run_filing_forecast,
    run_chapter_forecast,
    run_risk_score_forecast,
    detect_state_anomalies,
    get_risk_heatmap_data,
)
from query_box import (
    load_schema,
    get_db_table_name,
    get_actual_database_schema,
    extract_sql_from_response,
    generate_sql_from_question,
    validate_sql_with_judge,
    clean_sql_for_whitespace,
    execute_sql_query,
    should_generate_insights,
    detect_chart_type,
    _make_query_cache_key,
    create_temporary_table_from_dataframe,
    drop_temporary_table,
    is_followup_question,
    _initialize_conversation_memory,
    _append_conversation_memory,
    _handle_user_query,
    render_query_box_tab,
    _load_and_clean_csv,
)



load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bankruptcy_genbi.log", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("bankruptcy_genbi")

st.set_page_config(
    page_title="RiskIntel Portal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=Outfit:wght=400;500;600;700&display=swap');

    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc !important;
        color: #334155;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1600px;
    }

    /* Title Styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 500;
        margin-bottom: 0.1rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Card Panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        text-align: left;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        color: #2563eb !important;
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #64748b !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Global Table Header Customization (HTML tables) */
    th, [class*="header"], [data-testid="stTable"] th, .dataframe th {
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #0f172a !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #2563eb !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1) !important;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        transform: translateY(-1px);
    }

    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        background-color: #f8fafc !important;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb !important;
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #f1f5f9 !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.01) !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Scrollbars and cleanups */
    .stAlert {
        border-radius: 12px !important;
    }

    hr {
        margin: 1.5rem 0 !important;
        border: 0;
        border-top: 1px solid #e2e8f0;
    }

    /* Sidebar Navigation Premium Light Gray Styling */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label {
        color: #1e293b !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    /* Style the radio menu options to look like clean tab buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        padding-top: 1rem;
        gap: 0.8rem !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.2rem !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        transform: translateY(-1px);
    }

    /* Target checked radio item */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2) !important;
    }

    /* Ensure label text inside the checked radio element is readable (white) */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    /* Hide the radio selection circle for pure tab look */
    [data-testid="stSidebar"] div[role="radiogroup"] label div[role="presentation"] {
        display: none !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        margin-left: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SCHEMA UTILITY FUNCTIONS
# =============================================================================

def get_actual_col(col_name, columns_set):
    """Resolve database column name with case-insensitivity and character mapping support."""
    # First check exact or case-insensitive match
    for col in columns_set:
        if col.lower() == col_name.lower():
            return col
            
    # Then check mapping if the user calls using old names but we have clean names
    old_to_new = {
        "ac_no": "account_number",
        "client": "client_name",
        "open_date": "open_date",
        "first_name": "first_name",
        "middle_name": "middle_name",
        "last_name": "last_name",
        "ssn": "ssn",
        "addl_1": "address_line_1",
        "addl_2": "address_line_2",
        "city": "city",
        "state": "state",
        "zipcode": "zipcode",
        "record_type": "record_type",
        "match_code": "match_code",
        "match_score": "match_score",
        "consumer_id": "consumer_id",
        "consumer_type": "consumer_type",
        "notification_no": "notification_number",
        "notice_type": "notice_type",
        "case_no": "case_number",
        "chapter": "chapter",
        "original chapter": "original_chapter",
        "original_chapter": "original_chapter",
        "date_filed": "date_filed",
        "conversion_date": "conversion_date",
        "status": "status",
        "status_date": "status_date",
        "pd_ssn": "pd_ssn",
        "pd_first_name": "pd_first_name",
        "pd_middle_name": "pd_middle_name",
        "pd_lastt_name": "pd_last_name",
        "pd_last_name": "pd_last_name",
        "pd_suffix": "pd_suffix",
        "pd_aka": "pd_aka",
        "pd_aka2": "pd_aka2",
        "pd_dba": "pd_dba",
        "pd_dba2": "pd_dba2",
        "pd_addl_1": "pd_address_line_1",
        "pd_addl_2": "pd_address_line_2",
        "pd_city": "pd_city",
        "pd_state": "pd_state",
        "pd_zipcode": "pd_zipcode",
        "pd_phone": "pd_phone",
        "sd_ssn": "sd_ssn",
        "sd_first_name": "sd_first_name",
        "sd_middle_name": "sd_middle_name",
        "sd_lastt_name": "sd_last_name",
        "sd_last_name": "sd_last_name",
        "sd_suffix": "sd_suffix",
        "sd_aka": "sd_aka",
        "sd_aka2": "sd_aka2",
        "sd_dba": "sd_dba",
        "sd_dba2": "sd_dba2",
        "sd_addl_1": "sd_address_line_1",
        "sd_addl_2": "sd_address_line_2",
        "sd_city": "sd_city",
        "sd_state": "sd_state",
        "sd_zipcode": "sd_zipcode",
        "sd_phone": "sd_phone",
        "judge_name": "judge_name",
        "prose_indicator": "prose_indicator",
        "attorny_ssn": "attorney_ssn",
        "attorny_first_name": "attorney_first_name",
        "attorny_middle_name": "attorney_middle_name",
        "attorny_lastt_name": "attorney_last_name",
        "attorny_suffix": "attorney_suffix",
        "attorny_aka": "attorney_aka",
        "attorny_aka2": "attorney_aka2",
        "attorny_dba": "attorney_dba",
        "attorny_dba2": "attorney_dba2",
        "attorny_addl_1": "attorney_address_line_1",
        "attorny_addl_2": "attorney_address_line_2",
        "attorny_city": "attorney_city",
        "attorny_state": "attorney_state",
        "attorny_zipcode": "attorney_zipcode",
        "attorny_phone": "attorney_phone",
        "creditor_time": "creditor_meeting_time",
        "creditor_meeting_time": "creditor_meeting_time",
        "creditor_meeting_date": "creditor_meeting_date",
        "creditor_addl_1": "creditor_address_line_1",
        "creditor_addl_2": "creditor_address_line_2",
        "creditor_city": "creditor_city",
        "creditor_state": "creditor_state",
        "creditor_zipcode": "creditor_zipcode",
        "creditor_phone": "creditor_phone",
        "trustee_name": "trustee_name",
        "trustee_addl_1": "trustee_address_line_1",
        "trustee_addl_2": "trustee_address_line_2",
        "trustee_city": "trustee_city",
        "trustee_state": "trustee_state",
        "trustee_zipcode": "trustee_zipcode",
        "trustee_phone": "trustee_phone",
        "court_id": "court_id",
        "court_district": "court_district",
        "court_addl_1": "court_address_line_1",
        "court_addl_2": "court_address_line_2",
        "court_city": "court_city",
        "court_state": "court_state",
        "court_zipcode": "court_zipcode",
        "court_phone": "court_phone",
        "confirmation_date": "confirmation_date",
        "asset_indicator": "asset_indicator",
        "before_open": "before_open",
        "sasf": "sasf",
        "active_status": "active_status",
        "disposition_text": "disposition_text",
        "poc_bar_date": "poc_bar_date",
    }
    
    # Try normalized matches
    norm_col_name = col_name.lower().replace(" ", "_")
    mapped_target = old_to_new.get(norm_col_name, norm_col_name)
    
    for col in columns_set:
        norm_col = col.lower().replace(" ", "_")
        mapped_col = old_to_new.get(norm_col, norm_col)
        if mapped_col == mapped_target:
            return col
            
    return None


def to_bold_unicode(text):
    """Convert typical alphanumeric characters into mathematical bold characters to enforce formatting."""
    normal_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_chars = (
        "\U0001D400\U0001D401\U0001D402\U0001D403\U0001D404\U0001D405\U0001D406\U0001D407"
        "\U0001D408\U0001D409\U0001D40A\U0001D40B\U0001D40C\U0001D40D\U0001D40E\U0001D40F"
        "\U0001D410\U0001D411\U0001D412\U0001D413\U0001D414\U0001D415\U0001D416\U0001D417"
        "\U0001D418\U0001D419"
        "\U0001D41A\U0001D41B\U0001D41C\U0001D41D\U0001D41E\U0001D41F\U0001D420\U0001D421"
        "\U0001D422\U0001D423\U0001D424\U0001D425\U0001D426\U0001D427\U0001D428\U0001D429"
        "\U0001D42A\U0001D42B\U0001D42C\U0001D42D\U0001D42E\U0001D42F\U0001D430\U0001D431"
        "\U0001D432\U0001D433"
        "\U0001D7CE\U0001D7CF\U0001D7D0\U0001D7D1\U0001D7D2\U0001D7D3\U0001D7D4\U0001D7D5"
        "\U0001D7D6\U0001D7D7"
    )
    trans = str.maketrans(normal_chars, bold_chars)
    return str(text).translate(trans)



def format_table(df):
    """Utility to convert DataFrame column names to bold, uppercase representations."""
    if df is not None and not df.empty:
        df = df.copy()
        df.columns = [to_bold_unicode(str(col).upper()) for col in df.columns]
    return df

# =============================================================================
# HELPER DATA RETRIEVER FUNCTIONS (Imported from query_box.py)
# =============================================================================



# =============================================================================
# DYNAMIC METRICS & CHARTS RETRIEVER FOR DASHBOARD TABS
# =============================================================================

def get_dashboard_metrics(state_filter=None, chapter_filter=None, status_filter=None, 
                        prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                        date_start=None, date_end=None):
    """Query dynamic metrics for transactional dashboard tab with robust schema fallbacks"""
    fallback_metrics = {
        "total": 0,
        "active": 0,
        "converted": 0,
        "pro_se": 0,
        "avg_score": 0.0
    }
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            conn.close()
            return fallback_metrics

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        status_col = get_actual_col("status", columns) or get_actual_col("Active_Status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_dm = get_actual_col("client", columns)
        if client_filter and client_col_dm:
            where_clauses.append(f"{client_col_dm} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
        total_cases = cursor.fetchone()[0]

        active_cases = 0
        if status_col:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE LOWER({status_col}) IN ('active', 'open', 'y', '1') AND {where_clause}")
            active_cases = cursor.fetchone()[0]
        else:
            active_cases = total_cases

        converted_cases = 0
        orig_ch_col = get_actual_col("original_chapter", columns)
        if chapter_col and orig_ch_col:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {chapter_col} != {orig_ch_col} AND {orig_ch_col} IS NOT NULL AND {orig_ch_col} != '' AND {where_clause}")
            converted_cases = cursor.fetchone()[0]

        pro_se_cases = 0
        if prose_col:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE ({prose_col} = 1 OR LOWER({prose_col}) IN ('true', '1', 'y', 'yes')) AND {where_clause}")
            pro_se_cases = cursor.fetchone()[0]

        avg_score = 0.0
        score_col = get_actual_col("match_score", columns)
        if score_col:
            cursor.execute(f"SELECT AVG({score_col}) FROM {table_name} WHERE {where_clause}")
            avg_score = cursor.fetchone()[0] or 0.0

        conn.close()
        return {
            "total": total_cases,
            "active": active_cases,
            "converted": converted_cases,
            "pro_se": pro_se_cases,
            "avg_score": round(avg_score, 1)
        }
    except Exception as e:
        logger.error(f"Error querying dashboard metrics: {e}")
        return fallback_metrics


def get_chart_data(state_filter=None, chapter_filter=None, status_filter=None, 
                  prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                  date_start=None, date_end=None):
    """Query distribution data for dashboard visual widgets with dynamic fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")

        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_cd = get_actual_col("client", columns)
        if client_filter and client_col_cd:
            where_clauses.append(f"{client_col_cd} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        base_where = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_chapter = pd.DataFrame()
        if chapter_col:
            df_chapter = pd.read_sql_query(
                f"SELECT {chapter_col} as Chapter, COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND {chapter_col} IS NOT NULL AND {chapter_col} != '' GROUP BY {chapter_col}", conn
            )

        df_client = pd.DataFrame()
        client_col = None
        for col in ["record_type", "consumer_type", "notice_type", "client"]:
            actual = get_actual_col(col, columns)
            if actual:
                client_col = actual
                break
        if client_col:
            df_client = pd.read_sql_query(
                f"SELECT {client_col} as [Record Type], COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND {client_col} IS NOT NULL AND {client_col} != '' GROUP BY {client_col}", conn
            )

        df_state = pd.DataFrame()
        if state_col:
            df_state = pd.read_sql_query(
                f"SELECT {state_col} as State, COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND {state_col} IS NOT NULL AND {state_col} != '' GROUP BY {state_col} ORDER BY Count DESC LIMIT 10", conn
            )

        df_trend = pd.DataFrame()
        if date_col:
            df_trend = pd.read_sql_query(
                f""" SELECT substr({date_col}, 1, 4) as Year, COUNT(*) as Count
                FROM {table_name}
                WHERE ({base_where}) AND {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                GROUP BY Year
                ORDER BY Year
                """, conn
            )

        conn.close()
        return df_chapter, df_client, df_state, df_trend
    except Exception as e:
        logger.error(f"Error querying chart data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def get_case_status_distribution(state_filter=None, chapter_filter=None, status_filter=None, 
                                prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                                date_start=None, date_end=None):
    """Get case status distribution for pie chart visualization"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        state_col = get_actual_col("state", columns)
        chapter_col = get_actual_col("chapter", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_cs = get_actual_col("client", columns)
        if client_filter and client_col_cs:
            where_clauses.append(f"{client_col_cs} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_status = pd.DataFrame()
        if status_col:
            df_status = pd.read_sql_query(
                f"SELECT {status_col} as [Status], COUNT(*) as Count FROM {table_name} WHERE {status_col} IS NOT NULL AND {status_col} != '' AND {where_clause} GROUP BY {status_col} ORDER BY Count DESC",
                conn
            )
        else:
            if chapter_col:
                df_status = pd.read_sql_query(
                    f"SELECT {chapter_col} as [Status], COUNT(*) as Count FROM {table_name} WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' AND {where_clause} GROUP BY {chapter_col} ORDER BY Count DESC",
                    conn
                )
        
        conn.close()
        return df_status
    except Exception as e:
        logger.error(f"Error getting case status distribution: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_open_date_range():
    """Return (min_date, max_date) as datetime.date objects from the Open_date column."""
    import datetime
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}
        date_col = get_actual_col("Open_date", columns)
        if not date_col:
            conn.close()
            return None, None
        cursor.execute(
            f"SELECT MIN({date_col}), MAX({date_col}) FROM {table_name} "
            f"WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'"
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0] and row[1]:
            min_d = datetime.date.fromisoformat(str(row[0])[:10])
            max_d = datetime.date.fromisoformat(str(row[1])[:10])
            return min_d, max_d
        return None, None
    except Exception as e:
        logger.error(f"Error getting date range: {e}")
        return None, None


def get_monthly_filing_trends(state_filter=None, chapter_filter=None, status_filter=None,
                              prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                              date_start=None, date_end=None):
    """Monthly filing trends grouped by year and status for the drill-down analytics panel."""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_mt = get_actual_col("client", columns)
        if client_filter and client_col_mt:
            where_clauses.append(f"{client_col_mt} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        base_where = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_yearly = pd.DataFrame()
        if date_col:
            if status_col:
                df_yearly = pd.read_sql_query(
                    f""" SELECT substr({date_col}, 1, 4) as Year,
                           {status_col} as Status,
                           COUNT(*) as [Filing Count]
                    FROM {table_name}
                    WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                          AND ({base_where})
                    GROUP BY Year, Status
                    ORDER BY Year
                    """, conn
                )
            else:
                df_yearly = pd.read_sql_query(
                    f""" SELECT substr({date_col}, 1, 4) as Year,
                           COUNT(*) as [Filing Count]
                    FROM {table_name}
                    WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                          AND ({base_where})
                    GROUP BY Year
                    ORDER BY Year
                    """, conn
                )

        df_chapter_year = pd.DataFrame()
        if date_col and chapter_col:
            df_chapter_year = pd.read_sql_query(
                f""" SELECT substr({date_col}, 1, 4) as Year,
                       {chapter_col} as Chapter,
                       COUNT(*) as [Filing Count]
                FROM {table_name}
                WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                      AND {chapter_col} IS NOT NULL AND ({base_where})
                GROUP BY Year, Chapter
                ORDER BY Year, Chapter
                """, conn
            )

        conn.close()
        return df_yearly, df_chapter_year
    except Exception as e:
        logger.error(f"Error in monthly filing trends: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# BANKRUPTCY DATA INTELLIGENCE MODULES
# =============================================================================

def get_advanced_chapter_conversion_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                                            prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                                            date_start=None, date_end=None):
    """Advanced chapter analysis with risk metrics, conversion flows, and distribution insights"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_ac = get_actual_col("client", columns)
        if client_filter and client_col_ac:
            where_clauses.append(f"{client_col_ac} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_chapter_risk = pd.DataFrame()
        score_col = get_actual_col("match_score", columns)
        if chapter_col:
            if score_col:
                df_chapter_risk = pd.read_sql_query(
                    f""" SELECT 
                        {chapter_col} as [Chapter],
                        COUNT(*) as [Total Cases],
                        ROUND(AVG({score_col}), 1) as [Avg Risk Score],
                        ROUND(MIN({score_col}), 1) as [Min Risk],
                        ROUND(MAX({score_col}), 1) as [Max Risk],
                        SUM(CASE WHEN {score_col} >= 80 THEN 1 ELSE 0 END) as [High Risk Cases]
                    FROM {table_name}
                    WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' AND {where_clause}
                    GROUP BY {chapter_col}
                    ORDER BY [Total Cases] DESC
                    """, conn
                )
            else:
                df_chapter_risk = pd.read_sql_query(
                    f""" SELECT 
                        {chapter_col} as [Chapter],
                        COUNT(*) as [Total Cases],
                        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE {where_clause}), 1) as [% of Portfolio]
                    FROM {table_name}
                    WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' AND {where_clause}
                    GROUP BY {chapter_col}
                    ORDER BY [Total Cases] DESC
                    """, conn
                )

        df_conversions = pd.DataFrame()
        orig_ch_col = get_actual_col("original_chapter", columns)
        if chapter_col and orig_ch_col:
            df_conversions = pd.read_sql_query(
                f""" SELECT 
                    {orig_ch_col} as [From], 
                    {chapter_col} as [To], 
                    COUNT(*) as [Conversions],
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE {orig_ch_col} != {chapter_col} AND {orig_ch_col} IS NOT NULL AND {where_clause}), 1) as [% of Conversions]
                FROM {table_name} 
                WHERE {orig_ch_col} != {chapter_col} AND {orig_ch_col} IS NOT NULL AND {orig_ch_col} != '' AND {chapter_col} IS NOT NULL AND {chapter_col} != '' AND {where_clause}
                GROUP BY {orig_ch_col}, {chapter_col} 
                ORDER BY [Conversions] DESC
                LIMIT 5
                """, conn
            )

        df_chapter_demographics = pd.DataFrame()
        if chapter_col and prose_col:
            df_chapter_demographics = pd.read_sql_query(
                f""" SELECT 
                    {chapter_col} as [Chapter],
                    SUM(CASE WHEN {prose_col} IN ('Y', '1', 'true', 1) THEN 1 ELSE 0 END) as [Pro Se],
                    SUM(CASE WHEN {prose_col} NOT IN ('Y', '1', 'true', 1) THEN 1 ELSE 0 END) as [Represented]
                FROM {table_name}
                WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' AND {where_clause}
                GROUP BY {chapter_col}
                ORDER BY {chapter_col}
                """, conn
            )

        conn.close()
        return df_chapter_risk, df_conversions, df_chapter_demographics
    except Exception as e:
        logger.error(f"Error in advanced chapter conversion insights: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def get_legal_representation_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                                    prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                                    date_start=None, date_end=None):
    """Query representation data, pro se rates, top trustees and attorneys with dynamic check fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        state_col = get_actual_col("state", columns)
        chapter_col = get_actual_col("chapter", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_lr = get_actual_col("client", columns)
        if client_filter and client_col_lr:
            where_clauses.append(f"{client_col_lr} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_rep = pd.DataFrame()
        if prose_col:
            df_rep = pd.read_sql_query(
                f""" SELECT CASE WHEN {prose_col} = 1 OR LOWER({prose_col}) IN ('true', '1', 'y', 'yes') THEN 'Pro Se (Self-Represented)' ELSE 'Represented by Counsel' END as [Counsel Status],
                       COUNT(*) as [Case Count]
                FROM {table_name}
                WHERE {where_clause}
                GROUP BY [Counsel Status]
                """, conn
            )

        df_attorneys = pd.DataFrame()
        attorney_cols = [c for c in ["attorney_dba", "attorney_first_name", "attorney_last_name"] if get_actual_col(c, columns)]
        if len(attorney_cols) > 0:
            att_dba = get_actual_col("attorney_dba", columns)
            att_first = get_actual_col("attorney_first_name", columns)
            att_last = get_actual_col("attorney_last_name", columns)
            
            if att_dba and att_first and att_last:
                name_expr = f"COALESCE(NULLIF({att_dba}, ''), {att_first} || ' ' || {att_last})"
            elif att_dba:
                name_expr = att_dba
            elif att_first and att_last:
                name_expr = f"{att_first} || ' ' || {att_last}"
            elif att_last:
                name_expr = att_last
            else:
                name_expr = "''"

            df_attorneys = pd.read_sql_query(
                f""" SELECT {name_expr} as [Firm / Attorney Name], COUNT(*) as [Cases Handled]
                FROM {table_name}
                WHERE [Firm / Attorney Name] IS NOT NULL AND [Firm / Attorney Name] != '' AND [Firm / Attorney Name] != ' ' AND {where_clause}
                GROUP BY [Firm / Attorney Name]
                ORDER BY [Cases Handled] DESC
                LIMIT 10
                """, conn
            )

        df_trustees = pd.DataFrame()
        trustee_col = get_actual_col("trustee_name", columns)
        if trustee_col:
            df_trustees = pd.read_sql_query(
                f""" SELECT {trustee_col} as [Trustee Name], COUNT(*) as [Cases Administered]
                FROM {table_name}
                WHERE {trustee_col} IS NOT NULL AND {trustee_col} != '' AND {where_clause}
                GROUP BY {trustee_col}
                ORDER BY [Cases Administered] DESC
                LIMIT 10
                """, conn
            )
        conn.close()
        return df_rep, df_attorneys, df_trustees
    except Exception as e:
        logger.error(f"Error querying representation insights: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def get_client_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                       prose_filter=None, asset_filter=None, consumer_type_filter=None,
                       date_start=None, date_end=None):
    """Query client distribution, risk profiles, and case volume analytics"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        state_col = get_actual_col("state", columns)
        chapter_col = get_actual_col("chapter", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)
        client_col = get_actual_col("client", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_top_clients = pd.DataFrame()
        score_col = get_actual_col("match_score", columns)
        if client_col:
            if score_col:
                df_top_clients = pd.read_sql_query(
                    f""" SELECT 
                        {client_col} as [Client],
                        COUNT(*) as [Case Count],
                        ROUND(AVG({score_col}), 1) as [Avg Risk Score],
                        SUM(CASE WHEN {score_col} >= 80 THEN 1 ELSE 0 END) as [High Risk]
                    FROM {table_name}
                    WHERE {client_col} IS NOT NULL AND {client_col} != '' AND {client_col} != ' ' AND {where_clause}
                    GROUP BY {client_col}
                    ORDER BY [Case Count] DESC
                    LIMIT 8
                    """, conn
                )
            else:
                df_top_clients = pd.read_sql_query(
                    f""" SELECT 
                        {client_col} as [Client],
                        COUNT(*) as [Case Count]
                    FROM {table_name}
                    WHERE {client_col} IS NOT NULL AND {client_col} != '' AND {client_col} != ' ' AND {where_clause}
                    GROUP BY {client_col}
                    ORDER BY [Case Count] DESC
                    LIMIT 8
                    """, conn
                )

        df_client_chapter = pd.DataFrame()
        if client_col and chapter_col:
            df_client_chapter = pd.read_sql_query(
                f""" SELECT 
                    {client_col} as [Client],
                    {chapter_col} as [Chapter],
                    COUNT(*) as [Cases]
                FROM {table_name}
                WHERE {client_col} IS NOT NULL AND {client_col} != '' AND {chapter_col} IS NOT NULL AND {where_clause}
                GROUP BY {client_col}, {chapter_col}
                ORDER BY [Cases] DESC
                LIMIT 10
                """, conn
            )

        conn.close()
        return df_top_clients, df_client_chapter
    except Exception as e:
        logger.error(f"Error querying client insights: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_asset_and_geo_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                              prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                              date_start=None, date_end=None):
    """Query asset indicators and geographical trends with dynamic column check fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        state_col = get_actual_col("state", columns)
        chapter_col = get_actual_col("chapter", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)
        city_col = get_actual_col("city", columns)

        where_clauses = []
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_ag = get_actual_col("client", columns)
        if client_filter and client_col_ag:
            where_clauses.append(f"{client_col_ag} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_assets = pd.DataFrame()
        if asset_col:
            df_assets = pd.read_sql_query(
                f""" SELECT CASE WHEN {asset_col} = 1 OR LOWER({asset_col}) IN ('true', '1', 'y', 'yes') THEN 'Asset Cases' ELSE 'No-Asset Cases' END as [Liquidation Type],
                       COUNT(*) as [Case Count]
                FROM {table_name}
                WHERE {where_clause}
                GROUP BY [Liquidation Type]
                """, conn
            )

        df_geo = pd.DataFrame()
        if city_col and state_col:
            df_geo = pd.read_sql_query(
                f""" SELECT {city_col} || ', ' || {state_col} as [Jurisdiction], COUNT(*) as [Filings Density]
                FROM {table_name}
                WHERE {city_col} IS NOT NULL AND {city_col} != '' AND {state_col} IS NOT NULL AND {state_col} != '' AND {where_clause}
                GROUP BY [Jurisdiction]
                ORDER BY [Filings Density] DESC
                LIMIT 10
                """, conn
            )
        elif state_col:
            df_geo = pd.read_sql_query(
                f""" SELECT {state_col} as [Jurisdiction], COUNT(*) as [Filings Density]
                FROM {table_name}
                WHERE {state_col} IS NOT NULL AND {state_col} != '' AND {where_clause}
                GROUP BY [Jurisdiction]
                ORDER BY [Filings Density] DESC
                LIMIT 10
                """, conn
            )
            
        conn.close()
        return df_assets, df_geo
    except Exception as e:
        logger.error(f"Error querying asset and geo insights: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_predictive_data():
    """Query risk statistics and high-risk case listing with dynamic column check fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")

        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        score_col = get_actual_col("match_score", columns)
        df_scores = pd.DataFrame()
        if score_col:
            df_scores = pd.read_sql_query(
                f"SELECT {score_col} as [Match Score], COUNT(*) as Count FROM {table_name} WHERE {score_col} IS NOT NULL GROUP BY {score_col} ORDER BY {score_col}", conn
            )

        df_high_risk = pd.DataFrame()
        ac_col = get_actual_col("Ac_no", columns)
        if ac_col and score_col:
            first = get_actual_col("first_name", columns) or get_actual_col("pd_first_name", columns)
            last = get_actual_col("last_name", columns) or get_actual_col("pd_last_name", columns)
            
            if first and last:
                name_expr = f"{first} || ' ' || {last}"
            else:
                name_expr = ac_col

            state_col = get_actual_col("State", columns) or get_actual_col("PD_State", columns) or "NULL"
            chapter_col = get_actual_col("chapter", columns) or "NULL"
            status_col = get_actual_col("status", columns) or "NULL"

            df_high_risk = pd.read_sql_query(
                f""" SELECT {ac_col} as [Account No], {name_expr} as Name,
                       {state_col} as State, {chapter_col} as Chapter, {score_col} as [Match Score], {status_col} as Status
                FROM {table_name}
                WHERE {score_col} >= 95
                ORDER BY {score_col} DESC, {ac_col} ASC
                """, conn
            )
        conn.close()
        return df_scores, df_high_risk
    except Exception as e:
        logger.error(f"Error querying predictive data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# FORECASTING IMPROVEMENTS: TIME SERIES FORECASTING & EARLY WARNINGS
# =============================================================================

def get_timeseries_forecast_channels():
    """Time series forecast by channel with proactive planning capabilities"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        df_forecast = pd.DataFrame()
        
        channel_cols = [c for c in ["channel", "source", "record_type", "consumer_type", "notice_type"] if get_actual_col(c, columns)]
        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)
        score_col = get_actual_col("match_score", columns)
        score_expr = score_col if score_col else "100"
        
        if channel_cols and date_col:
            channel_col = get_actual_col(channel_cols[0], columns)
            df_forecast = pd.read_sql_query(
                f""" SELECT {channel_col} as Channel, 
                       substr({date_col}, 1, 4) as Year, 
                       COUNT(*) as [Filing Count],
                       AVG(CAST({score_expr} AS FLOAT)) as [Avg Risk Score]
                FROM {table_name}
                WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                GROUP BY {channel_col}, Year
                ORDER BY {channel_col}, Year
                """, conn
            )
        
        conn.close()
        return df_forecast
    except Exception as e:
        logger.error(f"Error in time series channel forecast: {e}")
        return pd.DataFrame()


def get_timeseries_forecast_sources():
    """Time series forecast by source with proactive planning capabilities"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        df_forecast = pd.DataFrame()
        
        source_cols = [c for c in ["source", "client", "originating_source", "filing_source"] if get_actual_col(c, columns)]
        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)
        
        if source_cols and date_col:
            source_col = get_actual_col(source_cols[0], columns)
            df_forecast = pd.read_sql_query(
                f""" SELECT {source_col} as Source, 
                       substr({date_col}, 1, 4) as Year, 
                       COUNT(*) as [Filing Count],
                       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY Year) * 100 as [Market Share %]
                FROM {table_name}
                WHERE {date_col} IS NOT NULL AND {date_col} != '' AND {date_col} LIKE '20%'
                       AND {source_col} IS NOT NULL AND {source_col} != ''
                GROUP BY {source_col}, Year
                ORDER BY Year DESC, [Filing Count] DESC
                """, conn
            )
        
        conn.close()
        return df_forecast
    except Exception as e:
        logger.error(f"Error in time series source forecast: {e}")
        return pd.DataFrame()


def get_early_warning_alerts(date_range_start=None, date_range_end=None):
    """Early warning radar system with analysis filter"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)
        score_col = get_actual_col("match_score", columns)
        chapter_col = get_actual_col("chapter", columns)

        alerts_data = {}
        
        if score_col and date_col:
            risk_threshold = 80
            date_filter = ""
            if date_range_start and date_range_end:
                date_filter = f" AND {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_high_risk_clusters = pd.read_sql_query(
                f""" SELECT COUNT(*) as [High Risk Cases], 
                       ROUND(AVG({score_col}), 2) as [Avg Risk Level],
                       COUNT(*) * 100.0 / COALESCE((SELECT COUNT(*) FROM {table_name} WHERE {date_col} IS NOT NULL AND {date_col} != '' {date_filter}), 1) as [% of Total]
                FROM {table_name}
                WHERE {score_col} >= {risk_threshold} AND {date_col} IS NOT NULL AND {date_col} != '' {date_filter}
                """, conn
            )
            alerts_data['high_risk'] = df_high_risk_clusters
        
        if date_col:
            date_filter = ""
            if date_range_start and date_range_end:
                date_filter = f" WHERE {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_spike_detection = pd.read_sql_query(
                f""" SELECT substr({date_col}, 1, 7) as Month, 
                       COUNT(*) as [Monthly Filings],
                       LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7)) as [Previous Month],
                       ROUND((COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7))) * 100.0 / 
                             NULLIF(LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7)), 0), 2) as [% Change]
                FROM {table_name}
                WHERE {date_col} IS NOT NULL AND {date_col} != '' {date_filter.replace('WHERE', 'AND') if date_filter else ''}
                GROUP BY Month
                ORDER BY Month DESC
                LIMIT 12
                """, conn
            )
            alerts_data['spike_detection'] = df_spike_detection
        
        if chapter_col and date_col:
            date_filter = ""
            if date_range_start and date_range_end:
                date_filter = f" AND {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_chapter_anomalies = pd.read_sql_query(
                f""" SELECT {chapter_col} as chapter, COUNT(*) as [Case Count],
                       ROUND(COUNT(*) * 100.0 / COALESCE((SELECT COUNT(*) FROM {table_name} WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' {date_filter}), 1), 2) as [% Distribution]
                FROM {table_name}
                WHERE {chapter_col} IS NOT NULL AND {chapter_col} != '' {date_filter}
                GROUP BY {chapter_col}
                ORDER BY [Case Count] DESC
                """, conn
            )
            alerts_data['chapter_anomalies'] = df_chapter_anomalies
        
        conn.close()
        return alerts_data
    except Exception as e:
        logger.error(f"Error in early warning system: {e}")
        return {}


# =============================================================================
# ADVANCED DRILL-DOWN & FILTERING FUNCTIONS
# =============================================================================

def get_filtered_cases(state_filter=None, chapter_filter=None, status_filter=None, 
                       prose_filter=None, asset_filter=None, consumer_type_filter=None, client_filter=None,
                       date_start=None, date_end=None, limit=1000):
    """Retrieve filtered case details for drill-down analysis"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)
        asset_col = get_actual_col("Asset_indicator", columns)
        consumer_col = get_actual_col("consumer_type", columns)
        date_col = get_actual_col("Open_date", columns)
        score_col = get_actual_col("match_score", columns) or "0"
        ac_col = get_actual_col("Ac_no", columns) or "Ac_no"

        first = get_actual_col("first_name", columns) or get_actual_col("pd_first_name", columns)
        last = get_actual_col("last_name", columns) or get_actual_col("pd_last_name", columns)
        if first and last:
            name_expr = f"{first} || ' ' || {last}"
        else:
            name_expr = ac_col

        where_clauses = []
        
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_fc = get_actual_col("client", columns)
        if client_filter and client_col_fc:
            where_clauses.append(f"{client_col_fc} = '{client_filter}'")
        if asset_filter and asset_col:
            where_clauses.append(f"{asset_col} = '{asset_filter}'")
        if consumer_type_filter and consumer_col:
            where_clauses.append(f"{consumer_col} = '{consumer_type_filter}'")
        if date_start and date_end and date_col:
            where_clauses.append(f"{date_col} BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f""" SELECT {ac_col} as [Case #], {name_expr} as Name, 
               COALESCE({state_col or 'NULL'}, 'N/A') as State, COALESCE({chapter_col or 'NULL'}, 'N/A') as Chapter,
               COALESCE({status_col or 'NULL'}, 'N/A') as Status, {date_col or 'NULL'} as [Open Date],
               COALESCE({score_col}, 0) as [Risk Score],
               CASE WHEN {prose_col or 'NULL'} IN ('Y', '1', 'true', 1) THEN 'Pro Se' ELSE 'Represented' END as [Representation],
               CASE WHEN {asset_col or 'NULL'} IN ('Y', '1', 'true', 1) THEN 'Asset' ELSE 'No Asset' END as [Asset Status],
               COALESCE({consumer_col or 'NULL'}, 'N/A') as [Consumer Type]
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY {score_col} DESC, {date_col or 'NULL'} DESC
        LIMIT {limit}
        """
        
        df_filtered = pd.read_sql_query(query, conn)
        conn.close()
        return df_filtered
    except Exception as e:
        logger.error(f"Error filtering cases: {e}")
        return pd.DataFrame()


def get_case_detail_by_id(case_number):
    """Get detailed information for a specific case"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}
        ac_col = get_actual_col("Ac_no", columns) or "Ac_no"
        
        query = f"SELECT * FROM {table_name} WHERE {ac_col} = '{case_number}' OR {ac_col} = {case_number} LIMIT 1"
        df_detail = pd.read_sql_query(query, conn)
        conn.close()
        return df_detail
    except Exception as e:
        logger.error(f"Error retrieving case detail: {e}")
        return pd.DataFrame()


# =============================================================================
# DETAILED TREND DRILL-DOWN QUANTITATIVE ANALYSIS
# =============================================================================

def get_trend_drilldown_insights(selected_year, state_filter=None, chapter_filter=None, status_filter=None, prose_filter=None, client_filter=None):
    """Fetch structured insights and breakdown data for a chosen year point"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        date_col = get_actual_col("Open_date", columns) or get_actual_col("date_filed", columns)
        if not date_col:
            conn.close()
            return None

        state_col = get_actual_col("state", columns)
        chapter_col = get_actual_col("chapter", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)

        where_clauses = [f"substr({date_col}, 1, 4) = '{selected_year}'"]
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if chapter_filter and chapter_col:
            where_clauses.append(f"{chapter_col} = {chapter_filter}")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_td = get_actual_col("client", columns)
        if client_filter and client_col_td:
            where_clauses.append(f"{client_col_td} = '{client_filter}'")

        where_clause = " AND ".join(where_clauses)

        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
        total_cases = cursor.fetchone()[0]

        df_chapters = pd.DataFrame()
        if chapter_col:
            df_chapters = pd.read_sql_query(
                f"SELECT {chapter_col} as Chapter, COUNT(*) as Count, ROUND(COUNT(*) * 100.0 / {total_cases}, 1) as Percentage FROM {table_name} WHERE {where_clause} GROUP BY {chapter_col}", conn
            )

        df_rep = pd.DataFrame()
        if prose_col:
            df_rep = pd.read_sql_query(
                f"SELECT CASE WHEN {prose_col} IN ('Y', '1', 'true', 1) THEN 'Pro Se' ELSE 'Represented' END as [Counsel Status], COUNT(*) as Count FROM {table_name} WHERE {where_clause} GROUP BY [Counsel Status]", conn
            )

        df_attorneys = pd.DataFrame()
        att_dba = get_actual_col("attorney_dba", columns)
        att_first = get_actual_col("attorney_first_name", columns)
        att_last = get_actual_col("attorney_last_name", columns)
        if att_dba or att_last:
            name_expr = f"COALESCE(NULLIF({att_dba}, ''), {att_first} || ' ' || {att_last})" if att_first else f"COALESCE({att_dba}, '')"
            df_attorneys = pd.read_sql_query(
                f"SELECT {name_expr} as Attorney, COUNT(*) as Count FROM {table_name} WHERE {where_clause} AND Attorney IS NOT NULL AND Attorney != '' GROUP BY Attorney ORDER BY Count DESC LIMIT 5", conn
            )

        conn.close()
        return {
            "total": total_cases,
            "chapters": df_chapters,
            "representation": df_rep,
            "attorneys": df_attorneys
        }
    except Exception as e:
        logger.error(f"Error fetching trend drilldown: {e}")
        return None


# =============================================================================
# INTERACTIVE CHAPTER DRILL-DOWN SUB-QUERIES
# =============================================================================

def get_chapter_drilldown_focus(selected_chapter, state_filter=None, status_filter=None, prose_filter=None, client_filter=None):
    """Retrieve regional concentration and status distribution for a specified chapter type"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        chapter_col = get_actual_col("chapter", columns)
        state_col = get_actual_col("state", columns)
        city_col = get_actual_col("city", columns)
        status_col = get_actual_col("status", columns)
        prose_col = get_actual_col("prose_indicator", columns)

        if not chapter_col:
            conn.close()
            return pd.DataFrame(), pd.DataFrame()

        where_clauses = [f"{chapter_col} = {selected_chapter}"]
        if state_filter and state_col:
            where_clauses.append(f"{state_col} = '{state_filter}'")
        if status_filter and status_col:
            where_clauses.append(f"{status_col} = '{status_filter}'")
        if prose_filter and prose_col:
            where_clauses.append(f"{prose_col} = '{prose_filter}'")
        client_col_ch = get_actual_col("client", columns)
        if client_filter and client_col_ch:
            where_clauses.append(f"{client_col_ch} = '{client_filter}'")

        where_clause = " AND ".join(where_clauses)

        df_geo = pd.DataFrame()
        if city_col and state_col:
            df_geo = pd.read_sql_query(
                f"""SELECT {city_col} || ', ' || {state_col} as Jurisdiction, COUNT(*) as Count 
                    FROM {table_name} 
                    WHERE {where_clause} AND {city_col} IS NOT NULL AND {city_col} != '' AND {city_col} != ' '
                    GROUP BY Jurisdiction 
                    ORDER BY Count DESC 
                    LIMIT 8""", conn
            )
        elif state_col:
            df_geo = pd.read_sql_query(
                f"""SELECT {state_col} as Jurisdiction, COUNT(*) as Count 
                    FROM {table_name} 
                    WHERE {where_clause} AND {state_col} IS NOT NULL AND {state_col} != '' AND {state_col} != ' '
                    GROUP BY Jurisdiction 
                    ORDER BY Count DESC 
                    LIMIT 8""", conn
            )

        df_status = pd.DataFrame()
        if status_col:
            df_status = pd.read_sql_query(
                f"""SELECT {status_col} as Status, COUNT(*) as Count 
                    FROM {table_name} 
                    WHERE {where_clause} AND {status_col} IS NOT NULL AND {status_col} != ''
                    GROUP BY {status_col} 
                    ORDER BY Count DESC""", conn
            )

        conn.close()
        return df_geo, df_status
    except Exception as e:
        logger.error(f"Error querying chapter drilldown parameters: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    logger.info("=== Session run initiated ===")

    # Initialize states
    if "data_in_db" not in st.session_state:
        st.session_state.data_in_db = False

    if "last_uploaded_file_name" not in st.session_state:
        st.session_state.last_uploaded_file_name = None

    if "actual_schema" not in st.session_state:
        if st.session_state.data_in_db:
            st.session_state.actual_schema = get_actual_database_schema()
        else:
            st.session_state.actual_schema = None

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = _initialize_conversation_memory()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": " **Welcome to GenBI Assistant!** "
            }
        ]

    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}

    if "temp_table_name" not in st.session_state:
        st.session_state.temp_table_name = None
    if "temp_table_schema" not in st.session_state:
        st.session_state.temp_table_schema = None
    if "temp_table_source_query" not in st.session_state:
        st.session_state.temp_table_source_query = None
    if "temp_table_dataframe" not in st.session_state:
        st.session_state.temp_table_dataframe = None

    # Load default schema
    schema = load_schema()
    if not schema:
        st.error("Fatal Error: Could not read schema configuration. Make sure schema.json is in the workspace.")
        return

    active_schema = st.session_state.actual_schema if st.session_state.actual_schema else schema

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    with st.sidebar:
        import os
        if os.path.exists("synchrony_logo.png"):
            st.image("synchrony_logo.png")
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; text-align: center;">
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                     Bankruptcy Risk Inteligence 
                </span>
            </div>
            <hr style="margin: 0.5rem 0 1.5rem 0 !important; border-top: 1px solid #cbd5e1;">
            """,
            unsafe_allow_html=True
        )
        if st.session_state.get("nav_redirect"):
            st.session_state.selected_tab = st.session_state.pop("nav_redirect")

        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "Analytics" if st.session_state.get("data_in_db", False) else "Data"

        selected_tab = st.radio(
            "Select Workspace",
            ["Data", "Analytics", "Forecasting", "Query Box"],
            key="selected_tab",
            label_visibility="collapsed"
        )


        # Analytics Filter Panel - frozen in sidebar
        if selected_tab == "Analytics" and st.session_state.get("data_in_db", False):
            st.markdown("---")
            st.markdown(
                """<div style="font-family:'Outfit',sans-serif;font-size:0.85rem;font-weight:700;
                color:#64748b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">
                Analytics Filters</div>""",
                unsafe_allow_html=True
            )
            _tname = get_db_table_name()
            _conn = sqlite3.connect("data.db")
            try:
                _states_df = pd.read_sql_query(
                    f"SELECT DISTINCT State FROM {_tname} WHERE State IS NOT NULL AND State != '' ORDER BY State",
                    _conn
                )
                _state_list = sorted(_states_df["State"].tolist())
            except:
                _state_list = []
            try:
                _chaps_df = pd.read_sql_query(
                    f"SELECT DISTINCT chapter FROM {_tname} WHERE chapter IS NOT NULL ORDER BY chapter",
                    _conn
                )
                _chapter_list = [int(c) for c in _chaps_df["chapter"].tolist() if str(c).isdigit()]
            except:
                _chapter_list = [7, 11, 13]
            try:
                _status_df = pd.read_sql_query(
                    f"SELECT DISTINCT status FROM {_tname} WHERE status IS NOT NULL AND status != '' ORDER BY status",
                    _conn
                )
                _status_list = sorted(_status_df["status"].tolist())
            except:
                _status_list = ["Active", "Closed", "Pending", "Converted", "Dismissed"]
            try:
                _client_df = pd.read_sql_query(
                    f"SELECT DISTINCT client FROM {_tname} WHERE client IS NOT NULL AND client != '' ORDER BY client",
                    _conn
                )
                _client_list = sorted(_client_df["client"].tolist())
            except:
                _client_list = []
            _conn.close()

            sel_state = st.selectbox("State / Territory", ["All States"] + _state_list, key="sb_state")
            state_filter = None if sel_state == "All States" else sel_state

            sel_chapter = st.selectbox("Bankruptcy Chapter", ["All Chapters"] + _chapter_list, key="sb_chapter")
            chapter_filter = None if sel_chapter == "All Chapters" else sel_chapter

            sel_status = st.selectbox("Case Status", ["All Statuses"] + _status_list, key="sb_status")
            status_filter = None if sel_status == "All Statuses" else sel_status

            sel_client = st.selectbox("Client", ["All Clients"] + _client_list, key="sb_client")
            client_filter = None if sel_client == "All Clients" else sel_client

            # Store in session state for use in the main body
            st.session_state["_analytics_filters"] = {
                "state_filter": state_filter,
                "chapter_filter": chapter_filter,
                "status_filter": status_filter,
                "client_filter": client_filter,
            }

        # Forecasting Filter Panel - frozen in sidebar
        if selected_tab == "Forecasting" and st.session_state.get("data_in_db", False):
            st.markdown("---")
            st.markdown(
                """<div style="font-family:'Outfit',sans-serif;font-size:0.85rem;font-weight:700;
                color:#64748b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">
                Forecast Filters</div>""",
                unsafe_allow_html=True
            )
            _ftname = get_db_table_name()
            _fconn = sqlite3.connect("data.db")
            try:
                _fchaps_df = pd.read_sql_query(
                    f"SELECT DISTINCT chapter FROM {_ftname} WHERE chapter IS NOT NULL ORDER BY chapter",
                    _fconn
                )
                _fc_chapter_list = [int(c) for c in _fchaps_df["chapter"].tolist() if str(c).isdigit()]
            except:
                _fc_chapter_list = [7, 11, 13]
            try:
                _fstates_df = pd.read_sql_query(
                    f"SELECT DISTINCT State FROM {_ftname} WHERE State IS NOT NULL AND State != '' ORDER BY State",
                    _fconn
                )
                _fc_state_list = sorted(_fstates_df["State"].tolist())
            except:
                _fc_state_list = []
            try:
                _fstatus_df = pd.read_sql_query(
                    f"SELECT DISTINCT status FROM {_ftname} WHERE status IS NOT NULL AND status != '' ORDER BY status",
                    _fconn
                )
                _fc_status_list = sorted(_fstatus_df["status"].tolist())
            except:
                _fc_status_list = ["Active", "Closed", "Pending", "Converted", "Dismissed"]

            # Query for date bounds
            _min_date_val = datetime.date(2020, 1, 1)
            _max_date_val = datetime.date(2026, 12, 31)
            try:
                cur = _fconn.cursor()
                cur.execute(f"PRAGMA table_info({_ftname})")
                cols = {r[1] for r in cur.fetchall()}
                _date_col = "Open_date" if "Open_date" in cols else ("date_filed" if "date_filed" in cols else None)
                if _date_col:
                    _df_minmax = pd.read_sql_query(
                        f"SELECT MIN({_date_col}) as min_d, MAX({_date_col}) as max_d FROM {_ftname} WHERE {_date_col} IS NOT NULL AND {_date_col} != '' AND {_date_col} LIKE '20%'",
                        _fconn
                    )
                    if not _df_minmax.empty:
                        min_str = _df_minmax["min_d"].iloc[0]
                        max_str = _df_minmax["max_d"].iloc[0]
                        if min_str:
                            _min_date_val = datetime.datetime.strptime(min_str[:10], "%Y-%m-%d").date()
                        if max_str:
                            _max_date_val = datetime.datetime.strptime(max_str[:10], "%Y-%m-%d").date()
            except Exception as e:
                pass

            _fconn.close()

            _fc_chapter_sel = st.selectbox("Chapter", ["All"] + _fc_chapter_list, key="fc_chapter")
            _fc_chapter_val = None if _fc_chapter_sel == "All" else int(_fc_chapter_sel)

            _fc_state_sel = st.selectbox("State / Territory", ["All"] + _fc_state_list, key="fc_state")
            _fc_state_val = None if _fc_state_sel == "All" else _fc_state_sel

            _fc_status_sel = st.selectbox("Case Status", ["All"] + _fc_status_list, key="fc_status")
            _fc_status_val = None if _fc_status_sel == "All" else _fc_status_sel

            _fc_prose_sel = st.selectbox(
                "Representation",
                ["All", "Pro Se (Self-Represented)", "Represented by Counsel"],
                key="fc_prose"
            )
            _fc_prose_val = None
            if _fc_prose_sel == "Pro Se (Self-Represented)":
                _fc_prose_val = "1"
            elif _fc_prose_sel == "Represented by Counsel":
                _fc_prose_val = "0"

            # Date range filter widget
            _fc_date_range = st.date_input(
                "Historical Date Range",
                value=(_min_date_val, _max_date_val),
                min_value=_min_date_val,
                max_value=_max_date_val,
                key="fc_date_range"
            )
            _fc_start_date = _min_date_val
            _fc_end_date = _max_date_val
            if isinstance(_fc_date_range, (tuple, list)) and len(_fc_date_range) == 2:
                _fc_start_date, _fc_end_date = _fc_date_range
            elif isinstance(_fc_date_range, (tuple, list)) and len(_fc_date_range) == 1:
                _fc_start_date = _fc_date_range[0]
                _fc_end_date = _max_date_val

            _fc_horizon_val = st.slider(" Forecast Horizon (months)", 6, 24, 12, key="fc_horizon")

            # Store in session state for use in the main body
            st.session_state["_forecast_filters"] = {
                "chapter_val": _fc_chapter_val,
                "state_val": _fc_state_val,
                "status_val": _fc_status_val,
                "prose_val": _fc_prose_val,
                "horizon": _fc_horizon_val,
                "start_date": _fc_start_date,
                "end_date": _fc_end_date,
            }

    st.markdown('<div class="subtitle">Enterprise Bankruptcy & Risk Intelligence Dashboard</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # WORKSPACE: DATA
    # -----------------------------------------------------------------
    if selected_tab == "Data":
        st.subheader("Repository Management")
        st.markdown("Upload transactional files dynamically to populate the SQLite analytics structure.")

        uploaded_file = st.file_uploader("Upload CSV database...", type=["csv"], help="Upload bankruptcy CSV data. Columns will be parsed dynamically.")

        if uploaded_file is not None:
            file_changed = uploaded_file.name != st.session_state.last_uploaded_file_name

            if file_changed:
                with st.spinner("Processing dataset..."):
                    df = _load_and_clean_csv(uploaded_file)

                    if df is not None:
                        conn = sqlite3.connect('data.db')
                        df.to_sql('uploaded_data', conn, if_exists='replace', index=False)
                        conn.close()

                        st.session_state.last_uploaded_file_name = uploaded_file.name
                        st.session_state.data_in_db = True
                        st.session_state.actual_schema = get_actual_database_schema()
                        st.session_state.nav_redirect = "Analytics"

                        st.success(f"Loaded: {uploaded_file.name}")
                        st.rerun()
                    else:
                        st.error("Failed to parse file. Ensure format is standard CSV.")

        if st.session_state.last_uploaded_file_name is not None and st.session_state.data_in_db:
            try:
                conn = sqlite3.connect("data.db")
                table_name = get_db_table_name()
                row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn)['count'].values[0]

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.2rem; border-radius: 12px; border: 1px solid #bbf7d0; margin-top: 1rem; margin-bottom: 1rem;">
                        <span style="color: #15803d; font-weight: 700; font-size: 1.25rem;"> Database Active</span>
                        <div style="color: #166534; font-size: 0.9rem; font-weight: 600; margin-top: 0.4rem;">
                            <strong>{row_count:,}</strong> records active in sqlite table: <code>{table_name}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander(" View Data Preview", expanded=True):
                    df_preview = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 50", conn)
                    st.dataframe(format_table(df_preview), width="stretch", hide_index=True)

                if st.session_state.actual_schema:
                    cols_list = [c['name'] for c in st.session_state.actual_schema['columns']]
                    with st.expander(f" Columns Dictionary ({len(cols_list)})", expanded=False):
                        for col in cols_list:
                            st.caption(f"{col}")

                conn.close()
            except Exception as e:
                logger.error(f"Error drawing data management cards: {e}")
                st.session_state.data_in_db = False

    # -----------------------------------------------------------------
    # WORKSPACE: ANALYTICS (OPTIMIZED INTERACTIVE DRILL-DOWN & CASE INSPECTOR)
    # -----------------------------------------------------------------
    elif selected_tab == "Analytics":
        # st.subheader("Case Portfolio Analytics")

        if not st.session_state.data_in_db:
            st.warning("No active dataset found. Please upload a CSV on the Data workspace to begin.")
        else:
            # Retrieve filters from session state (set by sidebar panel above)
            _filters = st.session_state.get("_analytics_filters", {})
            state_filter = _filters.get("state_filter")
            chapter_filter = _filters.get("chapter_filter")
            status_filter = _filters.get("status_filter")
            client_filter = _filters.get("client_filter")
            prose_filter = None  # kept for backward compat; not used in client-mode

            metrics = get_dashboard_metrics(
                state_filter=state_filter,
                chapter_filter=chapter_filter,
                status_filter=status_filter,
                prose_filter=prose_filter,
                client_filter=client_filter
            )

            #  Dynamic At-a-Glance Insight Banner 
            try:
                _scope_parts = []
                if client_filter:   _scope_parts.append(f"Client: <strong>{client_filter}</strong>")
                if state_filter:    _scope_parts.append(f"State: <strong>{state_filter}</strong>")
                if chapter_filter:  _scope_parts.append(f"Chapter: <strong>{chapter_filter}</strong>")
                if status_filter:   _scope_parts.append(f"Status: <strong>{status_filter}</strong>")
                _scope_label = " &nbsp;|&nbsp; ".join(_scope_parts) if _scope_parts else "<strong>All Cases &mdash; Full Portfolio View</strong>"

                _total     = metrics.get('total', 0) if metrics else 0
                _active    = metrics.get('active', 0) if metrics else 0
                _avg_score = metrics.get('avg_score', 0.0) if metrics else 0.0
                _active_pct = round((_active / _total * 100), 1) if _total > 0 else 0.0

                # Top chapter from live data
                try:
                    _tn = get_db_table_name()
                    _ic = sqlite3.connect("data.db")
                    _ch_col = get_actual_col("chapter", {r[1] for r in _ic.execute(f'PRAGMA table_info({_tn})').fetchall()})
                    _where_parts = []
                    if state_filter:   _where_parts.append(f"State = '{state_filter}'")
                    if chapter_filter: _where_parts.append(f"{_ch_col} = {chapter_filter}")
                    if status_filter:  _where_parts.append(f"status = '{status_filter}'")
                    if client_filter:  _where_parts.append(f"client = '{client_filter}'")
                    _wc = " AND ".join(_where_parts) if _where_parts else "1=1"
                    if _ch_col:
                        _top_ch_row = pd.read_sql_query(
                            f"SELECT {_ch_col} as ch, COUNT(*) as cnt FROM {_tn} WHERE {_wc} AND {_ch_col} IS NOT NULL GROUP BY {_ch_col} ORDER BY cnt DESC LIMIT 1",
                            _ic
                        )
                        _top_ch = f"Chapter {int(_top_ch_row.iloc[0]['ch'])}" if not _top_ch_row.empty else "N/A"
                        _top_ch_num = int(_top_ch_row.iloc[0]['ch']) if not _top_ch_row.empty else None
                    else:
                        _top_ch = "N/A"
                        _top_ch_num = None
                    _ic.close()
                except:
                    _top_ch = "N/A"
                    _top_ch_num = None

                # Build narrative text
                _context = f"Client <strong>{client_filter}</strong>" if client_filter else \
                           f"State <strong>{state_filter}</strong>" if state_filter else \
                           "the full portfolio"
                _active_health = "healthy active engagement" if _active_pct >= 50 else "lower active engagement - consider reviewing dormant cases"
                _ch_desc = {7: "liquidation (Chapter 7)", 11: "business reorganization (Chapter 11)",
                            13: "wage-earner repayment (Chapter 13)"}.get(_top_ch_num, f"Chapter {_top_ch_num}")
                _score_note = "above average - indicating strong data confidence" if _avg_score >= 70 \
                              else "moderate - data quality review may be beneficial" if _avg_score >= 40 \
                              else "below average - data completeness should be investigated"

                if client_filter or state_filter or chapter_filter or status_filter:
                    _narrative = (
                        f"The current selection shows <strong>{_total:,} cases</strong> across {_context}. "
                        f"Of these, <strong>{_active:,} ({_active_pct}%)</strong> are active, reflecting {_active_health}. "
                    )
                else:
                    _narrative = (
                        f"Your full portfolio contains <strong>{_total:,} bankruptcy cases</strong>. "
                        f"<strong>{_active:,} ({_active_pct}%)</strong> are currently active. "
                        f"Use the filters in the left panel to drill into a specific client, state, chapter, or status. "
                    )

                if _top_ch_num:
                    _narrative += f"The dominant filing type is <strong>{_ch_desc}</strong>, which represents the highest case concentration in this cohort. "
                if _avg_score > 0:
                    _narrative += f"The average portfolio match score stands at <strong>{_avg_score}%</strong>, which is {_score_note}."

                if status_filter:
                    _narrative += f" Results are scoped to <strong>{status_filter}</strong> cases only."

                st.markdown(
                    f"""
<div style="background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); border: 1px solid #c7d2fe; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: center; justify-content: space-between;">
<div style="flex: 1.6 1 420px; min-width: 300px;">
<div style="font-size:0.75rem; color:#6366f1; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.3rem;"> Executive Summery Insights</div>
<div style="font-size:0.82rem; color:#475569; margin-bottom:0.5rem;">{_scope_label}</div>
<div style="font-size:0.875rem; color:#1e293b; line-height:1.6; font-style: italic;">{_narrative}</div>
</div>
<div style="flex: 1 1 320px; min-width: 280px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
<div style="background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:0.4rem 0.6rem; text-align: center;">
<div style="font-size:0.65rem; color:#64748b; font-weight:600; text-transform:uppercase;">Total Cases</div>
<div style="font-size:1.15rem; font-weight:800; color:#1e293b;">{_total:,}</div>
</div>
<div style="background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:0.4rem 0.6rem; text-align: center;">
<div style="font-size:0.65rem; color:#64748b; font-weight:600; text-transform:uppercase;">Active Cases</div>
<div style="font-size:1.15rem; font-weight:800; color:#2563eb;">{_active:,} <span style="font-size:0.75rem; color:#64748b;">({_active_pct}%)</span></div>
</div>
<div style="background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:0.4rem 0.6rem; text-align: center;">
<div style="font-size:0.65rem; color:#64748b; font-weight:600; text-transform:uppercase;">Dominant Chapter</div>
<div style="font-size:1.15rem; font-weight:800; color:#7c3aed;">{_top_ch}</div>
</div>
<div style="background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:0.4rem 0.6rem; text-align: center;">
<div style="font-size:0.65rem; color:#64748b; font-weight:600; text-transform:uppercase;">Avg Match Score</div>
<div style="font-size:1.15rem; font-weight:800; color:#0f766e;">{_avg_score}%</div>
</div>
</div>
</div>
""",
                    unsafe_allow_html=True
                )
            except Exception as _e:
                logger.warning(f"Insight banner error: {_e}")
            # 



            panel_dashboard, panel_inspector = st.tabs([
                " High-Impact Analytics",
                " Case Inspector & Filtered Records"
            ])

            with panel_dashboard:
                col_chart1, col_chart2 = st.columns(2)

                # 1. Trajectory line chart with selection
                with col_chart1:
                    st.markdown("#### Case Volume Trend & Trajectory")
                    st.caption("Click any year point on the chart below to reveal deep-dive quantitative insights.")
                    _, _, _, df_trend = get_chart_data(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )
                    
                    selected_year_point = None
                    if df_trend is not None and not df_trend.empty:
                        df_trend["Year"] = df_trend["Year"].astype(str)
                        
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=df_trend["Year"], 
                            y=df_trend["Count"], 
                            mode='lines+markers',
                            line=dict(color='#2563eb', width=3),
                            marker=dict(size=10, color='#1e3a8a', symbol='circle'),
                            fill='tozeroy',
                            fillcolor='rgba(37, 99, 235, 0.08)',
                            hovertemplate='Year: %{x}<br>Cases: %{y}<extra></extra>'
                        ))
                        fig_trend.update_layout(
                            height=320,
                            margin=dict(l=20, r=20, t=10, b=20),
                            xaxis=dict(showgrid=False, type='category'),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='#f1f5f9',
                                tick0=0,
                                dtick=200,
                                range=[0, 1000],
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            clickmode='event+select'
                        )
                        
                        select_event = st.plotly_chart(
                            fig_trend, 
                            use_container_width=True, 
                            on_select="rerun", 
                            key="volume_trend_chart"
                        )
                        
                        if select_event and "selection" in select_event and select_event["selection"].get("points"):
                            selected_year_point = select_event["selection"]["points"][0].get("x")
                    else:
                        st.info("No trend trajectory parameters found in active cohort data.")

                # 2. Interactive Chapter Matrix with Selection
                with col_chart2:
                    st.markdown("#### Chapter and Risk Matrix")
                    st.caption("Filing volume categorized by chapter. Click any chapter bar to display regional concentrations and status distributions.")
                    df_chapter_risk, _, _ = get_advanced_chapter_conversion_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )
                    
                    selected_chapter_point = None
                    if df_chapter_risk is not None and not df_chapter_risk.empty:
                        df_chapter_risk['Chapter_Str'] = df_chapter_risk['Chapter'].astype(str)
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=df_chapter_risk['Chapter_Str'],
                            y=df_chapter_risk['Total Cases'],
                            marker_color='#1e3a8a',
                            hovertemplate='Chapter: %{x}<br>Cases: %{y}<extra></extra>'
                        ))
                        fig_bar.update_layout(
                            height=320,
                            margin=dict(l=20, r=20, t=10, b=20),
                            xaxis=dict(type='category', title="Chapter Variant"),
                            yaxis_title="Count of Filings",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            yaxis=dict(gridcolor='#f1f5f9'),
                            clickmode='event+select'
                        )
                        bar_select_event = st.plotly_chart(
                            fig_bar,
                            use_container_width=True,
                            on_select="rerun",
                            key="chapter_risk_matrix_chart"
                        )
                        if bar_select_event and "selection" in bar_select_event and bar_select_event["selection"].get("points"):
                            selected_chapter_point = bar_select_event["selection"]["points"][0].get("x")
                    else:
                        st.info("No chapter profile parameters found in active cohort data.")

                # Interactive Year Drill-down (from Trend Chart)
                if selected_year_point:
                    st.markdown(f"---")
                    st.markdown(f"###  Year Focus: {selected_year_point} Portfolio Insights")
                    drill_data = get_trend_drilldown_insights(
                        selected_year_point,
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )

                    if drill_data:
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            st.markdown(f"**Chapter Dynamics ({selected_year_point})**")
                            if not drill_data['chapters'].empty:
                                fig_ch_pie = go.Figure(data=[go.Pie(
                                    labels=drill_data['chapters']['Chapter'].astype(str),
                                    values=drill_data['chapters']['Count'],
                                    hole=0.3,
                                    marker=dict(colors=['#1e3a8a', '#2563eb', '#3b82f6', '#93c5fd'])
                                )])
                                fig_ch_pie.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
                                st.plotly_chart(fig_ch_pie, use_container_width=True)
                            else:
                                st.caption("No chapter dynamics returned for this selection.")

                        with dc2:
                            st.markdown(f"**Active Counsel Lead ({selected_year_point})**")
                            if not drill_data['attorneys'].empty:
                                st.dataframe(format_table(drill_data['attorneys']), hide_index=True, use_container_width=True)
                            else:
                                st.caption("No counsel listings discovered.")
                        
                        pro_se_pct = 0.0
                        if not drill_data['representation'].empty:
                            total_rep = drill_data['representation']['Count'].sum()
                            pro_se_df = drill_data['representation'][drill_data['representation']['Counsel Status'] == 'Pro Se']
                            if not pro_se_df.empty:
                                pro_se_pct = round((pro_se_df.iloc[0]['Count'] / total_rep) * 100, 1)

                        top_chapter_row = drill_data['chapters'].loc[drill_data['chapters']['Count'].idxmax()] if not drill_data['chapters'].empty else None
                        insight_text = f"**Analytical Assessment of Year {selected_year_point}:** "
                        if top_chapter_row is not None:
                            insight_text += f"Chapter {top_chapter_row['Chapter']} filings represent the primary segment, accounting for **{top_chapter_row['Percentage']}%** of the local portfolio. "
                        insight_text += f"The self-represented (Pro Se) rate is measured at **{pro_se_pct}%**."
                        st.markdown(insight_text)
                    st.markdown(f"---")

                # Interactive Chapter Drill-down (from Bar Chart)
                if selected_chapter_point:
                    st.markdown(f"---")
                    st.markdown(f"###  Chapter Focus: Chapter {selected_chapter_point} Dynamics")
                    
                    df_ch_geo, df_ch_status = get_chapter_drilldown_focus(
                        selected_chapter_point,
                        state_filter=state_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )

                    ch_col1, ch_col2 = st.columns(2)
                    with ch_col1:
                        st.markdown(f"**Geographic Hotspots within Chapter {selected_chapter_point}**")
                        if not df_ch_geo.empty:
                            fig_ch_geo = go.Figure(go.Bar(
                                x=df_ch_geo['Count'],
                                y=df_ch_geo['Jurisdiction'],
                                orientation='h',
                                marker_color='#2563eb'
                            ))
                            fig_ch_geo.update_layout(
                                height=280,
                                margin=dict(l=150, r=10, t=10, b=10),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                yaxis=dict(autorange="reversed")
                            )
                            st.plotly_chart(fig_ch_geo, use_container_width=True)
                        else:
                            st.info(f"No regional parameters found for Chapter {selected_chapter_point}.")

                    with ch_col2:
                        st.markdown(f"**Case Status within Chapter {selected_chapter_point}**")
                        if not df_ch_status.empty:
                            fig_ch_status = go.Figure(data=[go.Pie(
                                labels=df_ch_status['Status'],
                                values=df_ch_status['Count'],
                                hole=0.4,
                                marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#cbd5e1'])
                            )])
                            fig_ch_status.update_layout(
                                height=280,
                                margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h")
                            )
                            st.plotly_chart(fig_ch_status, use_container_width=True)
                        else:
                            st.info("No status indicators discovered.")
                    st.markdown(f"---")

                col_sec1, col_sec2 = st.columns(2)
                with col_sec1:
                    st.markdown("#### Top Debtor Attorneys and Concentrations")
                    _, df_attorneys, _ = get_legal_representation_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )
                    if df_attorneys is not None and not df_attorneys.empty:
                        st.dataframe(format_table(df_attorneys.head(8)), hide_index=True, use_container_width=True)
                    else:
                        st.info("No counselor or attorney fields discovered in active cohort.")

                with col_sec2:
                    st.markdown("#### Liquidation Profile Metrics")
                    df_assets, _ = get_asset_and_geo_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        client_filter=client_filter
                    )
                    if df_assets is not None and not df_assets.empty:
                        fig_donut = go.Figure(data=[go.Pie(
                            labels=df_assets['Liquidation Type'],
                            values=df_assets['Case Count'],
                            hole=.4,
                            marker=dict(colors=['#2563eb', '#cbd5e1'])
                        )])
                        fig_donut.update_layout(
                            height=250,
                            margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", y=-0.1)
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)
                    else:
                        st.info("No liquidation profile parameters found.")

            # -----------------------------------------------------------------
            # CASE INSPECTOR & FILTERED RECORDS
            # -----------------------------------------------------------------
            with panel_inspector:
                df_filtered_cases = get_filtered_cases(
                    state_filter=state_filter,
                    chapter_filter=chapter_filter,
                    status_filter=status_filter,
                    prose_filter=prose_filter,
                    client_filter=client_filter
                )

                if not df_filtered_cases.empty:
                    st.markdown("###  Filtered Case Summary")
                    
                    total_filtered = len(df_filtered_cases)
                    avg_risk_score = round(df_filtered_cases["Risk Score"].mean(), 1)
                    prose_cases_count = len(df_filtered_cases[df_filtered_cases["Representation"] == "Pro Se"])
                    prose_percentage = round((prose_cases_count / total_filtered) * 100, 1) if total_filtered > 0 else 0.0
                    
                    asset_cases_count = len(df_filtered_cases[df_filtered_cases["Asset Status"] == "Asset"])
                    asset_percentage = round((asset_cases_count / total_filtered) * 100, 1) if total_filtered > 0 else 0.0

                    info_col1, info_col2 = st.columns([1, 1.5])
                    with info_col1:
                        st.markdown(
                            f"""
                            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; height: 260px;">
                                <h5 style="margin-top: 0px; color: #1e293b;">Cohort Distribution Metrics</h5>
                                <p style="margin-bottom: 8px;"><b>Total Filtered Cases:</b> {total_filtered:,}</p>
                                <p style="margin-bottom: 8px;"><b>Average Risk Index:</b> {avg_risk_score}%</p>
                                <p style="margin-bottom: 8px;"><b>Pro Se Representation:</b> {prose_cases_count} cases ({prose_percentage}%)</p>
                                <p style="margin-bottom: 0px;"><b>Liquidation Assets Present:</b> {asset_cases_count} cases ({asset_percentage}%)</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with info_col2:
                        fig_ratios = go.Figure()
                        fig_ratios.add_trace(go.Bar(
                            name='Represented',
                            y=['Representation Profile'],
                            x=[100 - prose_percentage],
                            orientation='h',
                            marker_color='#10b981'
                        ))
                        fig_ratios.add_trace(go.Bar(
                            name='Pro Se',
                            y=['Representation Profile'],
                            x=[prose_percentage],
                            orientation='h',
                            marker_color='#ef4444'
                        ))
                        fig_ratios.add_trace(go.Bar(
                            name='No-Asset',
                            y=['Asset Profile'],
                            x=[100 - asset_percentage],
                            orientation='h',
                            marker_color='#cbd5e1'
                        ))
                        fig_ratios.add_trace(go.Bar(
                            name='Asset Cases',
                            y=['Asset Profile'],
                            x=[asset_percentage],
                            orientation='h',
                            marker_color='#2563eb'
                        ))
                        fig_ratios.update_layout(
                            barmode='stack',
                            height=260,
                            margin=dict(l=150, r=20, t=10, b=40),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis=dict(range=[0, 100], ticksuffix='%', gridcolor='#f1f5f9'),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.1)
                        )
                        st.plotly_chart(fig_ratios, use_container_width=True)

                    st.markdown("---")
                    st.markdown("###  Filtered Case Records")
                    st.dataframe(format_table(df_filtered_cases), hide_index=True, use_container_width=True)

                    csv_data = df_filtered_cases.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=" Export Filtered Cases as CSV",
                        data=csv_data,
                        file_name="filtered_bankruptcy_cases.csv",
                        mime="text/csv"
                    )

                    # Dynamic details lookups
                    st.markdown("---")
                    st.markdown("###  Case Profile Detail Inspector")
                    st.markdown("Select an individual Account Number from the dropdown below to view a detailed case dashboard.")
                    
                    case_choices = df_filtered_cases["Case #"].tolist()
                    selected_case_no = st.selectbox("Select Account/Case Number", case_choices, key="case_inspector_selector")

                    if selected_case_no:
                        case_detail_df = get_case_detail_by_id(selected_case_no)
                        if not case_detail_df.empty:
                            case_row = case_detail_df.iloc[0]
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            det_col1, det_col2 = st.columns([1, 2.5])
                            with det_col1:
                                score_field = get_actual_col('match_score', case_row.keys())
                                score = float(case_row.get(score_field, 0)) if score_field else 0.0
                                
                                fig_gauge = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = score,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "Assessment Index", 'font': {'size': 16, 'family': 'Outfit'}},
                                    gauge = {
                                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                                        'bar': {'color': "#1e3a8a"},
                                        'bgcolor': "white",
                                        'borderwidth': 2,
                                        'bordercolor': "#cbd5e1",
                                        'steps': [
                                            {'range': [0, 50], 'color': '#f8fafc'},
                                            {'range': [50, 80], 'color': '#eff6ff'},
                                            {'range': [80, 100], 'color': '#dbeafe'}
                                        ],
                                    }
                                ))
                                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=20))
                                st.plotly_chart(fig_gauge, use_container_width=True)

                            with det_col2:
                                card_c1, card_c2 = st.columns(2)
                                
                                first_name_f = get_actual_col('first_name', case_row.keys()) or get_actual_col('pd_first_name', case_row.keys())
                                last_name_f = get_actual_col('last_name', case_row.keys()) or get_actual_col('pd_last_name', case_row.keys())
                                state_f = get_actual_col('state', case_row.keys())
                                city_f = get_actual_col('city', case_row.keys())
                                ac_f = get_actual_col('Ac_no', case_row.keys())
                                
                                name_val = f"{case_row.get(first_name_f, '')} {case_row.get(last_name_f, '')}".strip() if (first_name_f and last_name_f) else "N/A"
                                
                                with card_c1:
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                                            <span style="font-size: 0.85rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Entity Attributes</span>
                                            <h4 style="margin: 0.2rem 0; color: #1e293b;">{name_val}</h4>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Account ID:</b> {case_row.get(ac_f, 'N/A')}</p>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Jurisdiction:</b> {case_row.get(city_f, 'N/A')}, {case_row.get(state_f, 'N/A')}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    chapter_f = get_actual_col('chapter', case_row.keys())
                                    status_f = get_actual_col('status', case_row.keys())
                                    trustee_f = get_actual_col('trustee_name', case_row.keys())
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8fafc; border-left: 4px solid #10b981; padding: 1rem; border-radius: 4px;">
                                            <span style="font-size: 0.85rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Procedural Status</span>
                                            <h4 style="margin: 0.2rem 0; color: #1e293b;">Chapter {case_row.get(chapter_f, 'N/A')}</h4>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Case Status:</b> {case_row.get(status_f, 'N/A')}</p>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Assigned Trustee:</b> {case_row.get(trustee_f, 'N/A')}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                with card_c2:
                                    open_f = get_actual_col('open_date', case_row.keys())
                                    status_date_f = get_actual_col('status_date', case_row.keys())
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8fafc; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                                            <span style="font-size: 0.85rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Timeline Coordinates</span>
                                            <h4 style="margin: 0.2rem 0; color: #1e293b;">Filing Chronology</h4>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Open Date:</b> {case_row.get(open_f, 'N/A')}</p>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Status Date:</b> {case_row.get(status_date_f, 'N/A')}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    client_f = get_actual_col('client_name', case_row.keys()) or get_actual_col('client', case_row.keys())
                                    att_dba_f = get_actual_col('attorney_dba', case_row.keys())
                                    prose_f = get_actual_col('prose_indicator', case_row.keys())
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8fafc; border-left: 4px solid #6366f1; padding: 1rem; border-radius: 4px;">
                                            <span style="font-size: 0.85rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Sponsorship & Counsel</span>
                                            <h4 style="margin: 0.2rem 0; color: #1e293b;">{case_row.get(client_f, 'Sponsor N/A')}</h4>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Attorney:</b> {case_row.get(att_dba_f, 'N/A')}</p>
                                            <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><b>Pro Se:</b> {'Yes' if str(case_row.get(prose_f)) in ['1', 'Y', 'true', 'yes'] else 'No'}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.warning("Could not load details for selected case.")
                else:
                    st.info("No records match the chosen parameters. Use the filters at the top to configure the cohort.")

    # -----------------------------------------------------------------
    # WORKSPACE: FORECASTING
    # -----------------------------------------------------------------
    elif selected_tab == "Forecasting":
        # st.subheader(" Filing Forecast & Risk Intelligence")
        # st.caption("Ensemble ML forecast (Linear, Polynomial, Exp Smoothing, Ridge) with business insights.")

        if not st.session_state.data_in_db:
            st.warning("No active dataset found. Please upload a CSV on the 'Data' workspace.")
        else:
            # Retrieve filters from sidebar session state
            _ff = st.session_state.get("_forecast_filters", {})
            fc_chapter_val = _ff.get("chapter_val")
            fc_state_val   = _ff.get("state_val")
            fc_status_val  = _ff.get("status_val")
            fc_prose_val   = _ff.get("prose_val")
            fc_horizon     = _ff.get("horizon", 12)
            fc_start_date  = _ff.get("start_date")
            fc_end_date    = _ff.get("end_date")

            #  Run models 
            @st.cache_data(ttl=120, show_spinner=False)
            def _fc(ch, st_val, stat_val, pr_val, h, start_d, end_d):
                df_s = load_monthly_series(
                    chapter_filter=ch,
                    state_filter=st_val,
                    status_filter=stat_val,
                    prose_filter=pr_val,
                    client_filter=None,
                    start_date=start_d,
                    end_date=end_d,
                )
                return run_filing_forecast(df_s, horizon_months=h), df_s

            with st.spinner("Running forecast..."):
                fc_result, df_series = _fc(
                    fc_chapter_val,
                    fc_state_val,
                    fc_status_val,
                    fc_prose_val,
                    fc_horizon,
                    fc_start_date,
                    fc_end_date,
                )

            if not fc_result:
                st.error("Not enough data to forecast. Try broadening your filter selections (e.g., selecting 'All' for Chapter/State/Status/etc.) or extending the date range.")
            else:
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Next 3 Months Filing Count", f"{fc_result['next_q']:,}")
                k2.metric("12-Month Projection Filing Count", f"{fc_result['next_year']:,}")
                k3.metric("Trend", fc_result["trend"].title(), f"{fc_result['trend_pct']:+.1f}%")
                _acc = fc_result.get("metrics", {}).get("Accuracy", 0.0)
                k4.metric("Model Accuracy", f"{_acc:.1f}%")
                k5.metric("Peak Month", fc_result["peak_month"])

                if "insights" in fc_result and fc_result["insights"]:
                    st.markdown(
                        """
                        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                                    border: 1px solid #bae6fd; border-radius: 12px;
                                    padding: 0.9rem 1.25rem; margin-top: 1rem; margin-bottom: 0.5rem;">
                            <div style="font-size:0.75rem; color:#0369a1; font-weight:700;
                                        letter-spacing:0.07em; text-transform:uppercase;
                                        margin-bottom:0.5rem;"> Key Forecasting Insights</div>
                            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.875rem; color: #1e293b; line-height: 1.6;">
                        """ + "".join([f"<li style='margin-bottom:0.4rem;'>{ins}</li>" for ins in fc_result["insights"][:4]]) + """
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                hist_df   = fc_result["history"]
                fcast_df  = fc_result["forecast"]
                fitted_df = fc_result.get("fitted", pd.DataFrame())

                # ── Load the FULL available series (no date filter) so we can overlay
                #    real observed values on the forecast period for backtesting. ──────
                @st.cache_data(ttl=120, show_spinner=False)
                def _load_full_series(ch, st_val, stat_val, pr_val):
                    return load_monthly_series(
                        chapter_filter=ch, state_filter=st_val,
                        status_filter=stat_val, prose_filter=pr_val,
                        client_filter=None, start_date=None, end_date=None,
                    )

                df_full = _load_full_series(
                    fc_chapter_val, fc_state_val, fc_status_val, fc_prose_val
                )

                # Find actual data that falls inside the forecast window
                df_actual_in_fcst = pd.DataFrame()
                if not df_full.empty and not fcast_df.empty and "Filing_Count" in df_full.columns:
                    fcst_start = fcast_df["Month"].iloc[0]
                    fcst_end   = fcast_df["Month"].iloc[-1]
                    df_actual_in_fcst = df_full[
                        (df_full["Month"] >= fcst_start) &
                        (df_full["Month"] <= fcst_end)
                    ][["Month", "Filing_Count"]].copy()

                has_actual_in_fcst = not df_actual_in_fcst.empty

                st.markdown(
                    f"""
                    <div style="font-size:0.8rem; font-weight:700; color:#0369a1;
                                text-transform:uppercase; letter-spacing:0.07em;
                                margin: 1rem 0 0.3rem 0;">
                         Actual vs Forecasted — Filing Count
                        {"&nbsp;<span style='background:#dcfce7; color:#15803d; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; vertical-align:middle;'>✔ ACTUAL DATA AVAILABLE IN FORECAST WINDOW</span>" if has_actual_in_fcst else ""}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                fig_avf = go.Figure()

                # ── 1. Actual historical filings (within selected date range) ──────────
                fig_avf.add_trace(go.Scatter(
                    x=hist_df["Month"], y=hist_df["Filing_Count"],
                    name="Actual Filings (Training Window)",
                    mode="lines+markers",
                    line=dict(color="#1e3a8a", width=2.5),
                    marker=dict(size=4, color="#1e3a8a"),
                    hovertemplate="%{x|%b %Y}  Actual: <b>%{y:,.0f}</b><extra></extra>"
                ))

                # ── 2. Model in-sample fitted line ────────────────────────────────────
                if not fitted_df.empty and "Fitted" in fitted_df.columns:
                    fig_avf.add_trace(go.Scatter(
                        x=fitted_df["Month"], y=fitted_df["Fitted"],
                        name="Model Fit (In-Sample)",
                        mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dot"),
                        hovertemplate="%{x|%b %Y}  Model Fit: <b>%{y:,.0f}</b><extra></extra>"
                    ))

                # ── 3. Ensemble forecast (future / forecast window) ───────────────────
                if "Ensemble Forecast" in fcast_df.columns:
                    fig_avf.add_trace(go.Scatter(
                        x=fcast_df["Month"], y=fcast_df["Ensemble Forecast"],
                        name="Ensemble Forecast",
                        mode="lines+markers",
                        line=dict(color="#dc2626", width=2.5),
                        marker=dict(size=5, symbol="diamond", color="#dc2626"),
                        hovertemplate="%{x|%b %Y}  Forecast: <b>%{y:,.0f}</b><extra></extra>"
                    ))

                # ── 4. Actual data within the forecast window (backtesting overlay) ───
                if has_actual_in_fcst:
                    fig_avf.add_trace(go.Scatter(
                        x=df_actual_in_fcst["Month"],
                        y=df_actual_in_fcst["Filing_Count"],
                        name="Actual Filings (Forecast Window)",
                        mode="lines+markers",
                        line=dict(color="#16a34a", width=2.5),
                        marker=dict(size=6, color="#16a34a", symbol="circle"),
                        hovertemplate="%{x|%b %Y}  Actual (observed): <b>%{y:,.0f}</b><extra></extra>"
                    ))

                    # Compute simple point-in-time accuracy for the overlap period
                    try:
                        merged_bk = pd.merge(
                            df_actual_in_fcst.rename(columns={"Filing_Count": "Actual"}),
                            fcast_df[["Month", "Ensemble Forecast"]].rename(columns={"Ensemble Forecast": "Predicted"}),
                            on="Month", how="inner"
                        )
                        if not merged_bk.empty:
                            mask = merged_bk["Actual"] > 0
                            if mask.any():
                                bk_mape = float(np.mean(
                                    np.abs((merged_bk.loc[mask, "Actual"] - merged_bk.loc[mask, "Predicted"])
                                           / merged_bk.loc[mask, "Actual"])
                                ) * 100)
                                bk_acc = max(0.0, 100.0 - bk_mape)
                            else:
                                bk_acc = None
                        else:
                            bk_acc = None
                    except Exception:
                        bk_acc = None
                else:
                    bk_acc = None

                # ── Vertical separator at training/forecast boundary ──────────────────
                fig_avf.add_vline(
                    x=hist_df["Month"].iloc[-1].timestamp() * 1000,
                    line_dash="dot", line_color="#94a3b8", line_width=1.5,
                    annotation_text="▶ Forecast Start", annotation_position="top right",
                    annotation_font=dict(color="#64748b", size=11)
                )

                # ── Shaded forecast region ────────────────────────────────────────────
                if "Ensemble Forecast" in fcast_df.columns and len(fcast_df) > 0:
                    fig_avf.add_vrect(
                        x0=fcast_df["Month"].iloc[0].timestamp() * 1000,
                        x1=fcast_df["Month"].iloc[-1].timestamp() * 1000,
                        fillcolor="rgba(220, 38, 38, 0.04)",
                        layer="below", line_width=0
                    )

                fig_avf.update_layout(
                    height=400, hovermode="x unified",
                    title=dict(
                        text=f" Monthly Filing Count — Actual vs Forecasted  ({fc_horizon}-Month Horizon)",
                        font=dict(size=14, color="#0f172a")
                    ),
                    xaxis_title=None, yaxis_title="Filings",
                    legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
                    margin=dict(l=50, r=30, t=70, b=40),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickformat="%b %Y"),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                )
                st.plotly_chart(fig_avf, use_container_width=True)

                # ── Backtesting accuracy callout (when actual data overlaps forecast) ─
                if has_actual_in_fcst and bk_acc is not None:
                    bk_color = "#15803d" if bk_acc >= 85 else "#d97706" if bk_acc >= 70 else "#dc2626"
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                                    border: 1px solid #86efac; border-radius: 10px;
                                    padding: 0.65rem 1.1rem; margin-bottom: 0.8rem;
                                    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                            <span style="font-size:0.8rem; font-weight:700; color:#166534;
                                         text-transform:uppercase; letter-spacing:0.05em;">
                                 Backtest Result
                            </span>
                            <span style="font-size:0.875rem; color:#1e293b;">
                                The model predicted <b>{len(df_actual_in_fcst)}</b> months where
                                actual data is now available.
                                Backtest accuracy:&nbsp;
                                <b style="color:{bk_color}; font-size:1rem;">{bk_acc:.1f}%</b>
                                &nbsp;(comparing forecast vs observed filings in that period).
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                _mape_val = fc_result.get("metrics", {}).get("MAPE (%)", 0)
                _mae_val  = fc_result.get("metrics", {}).get("MAE", 0)
                _rmse_val = fc_result.get("metrics", {}).get("RMSE", 0)
                _acc_color = "#15803d" if _acc >= 85 else "#d97706" if _acc >= 70 else "#dc2626"
                st.markdown(
                    f"""
                    <div style="display:flex; gap:1.2rem; margin-bottom:1rem; flex-wrap:wrap;">
                        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                                    padding:0.5rem 1rem; font-size:0.85rem;">
                            <span style="color:#64748b; font-weight:600;">Model Accuracy</span>&nbsp;
                            <span style="color:{_acc_color}; font-weight:700; font-size:1rem;">{_acc:.1f}%</span>
                        </div>
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                                    padding:0.5rem 1rem; font-size:0.85rem;">
                            <span style="color:#64748b; font-weight:600;">MAE</span>&nbsp;
                            <span style="color:#1e3a8a; font-weight:700;">{_mae_val:.1f}</span>
                        </div>
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                                    padding:0.5rem 1rem; font-size:0.85rem;">
                            <span style="color:#64748b; font-weight:600;">RMSE</span>&nbsp;
                            <span style="color:#1e3a8a; font-weight:700;">{_rmse_val:.1f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                df_chap_fc = run_chapter_forecast(df_series, horizon_months=fc_horizon)
                if not df_chap_fc.empty:
                    chap_cols    = [c for c in df_chap_fc.columns if c != "Month"]
                    chap_palette = {"Chapter 7": "#2563eb", "Chapter 11": "#16a34a", "Chapter 13": "#dc2626"}
                    fig_ch = go.Figure()
                    for col_h in [c for c in df_series.columns if c.startswith("Ch_")]:
                        label = col_h.replace("Ch_", "Chapter ")
                        fig_ch.add_trace(go.Scatter(
                            x=df_series["Month"], y=df_series[col_h],
                            name=label, mode="lines",
                            line=dict(color=chap_palette.get(label, "#64748b"), width=1.8),
                            hovertemplate=f"%{{x|%b %Y}}  {label}: %{{y}}<extra></extra>"
                        ))
                    for col_f in chap_cols:
                        fig_ch.add_trace(go.Scatter(
                            x=df_chap_fc["Month"], y=df_chap_fc[col_f],
                            name=f"{col_f} (fcst)", mode="lines",
                            line=dict(color=chap_palette.get(col_f, "#64748b"), width=2, dash="dash"),
                            hovertemplate=f"%{{x|%b %Y}}  {col_f}: %{{y:.0f}}<extra></extra>"
                        ))
                    fig_ch.add_vline(
                        x=df_series["Month"].iloc[-1].timestamp() * 1000,
                        line_dash="dot", line_color="#94a3b8"
                    )
                    fig_ch.update_layout(
                        height=320, title=" Chapter Breakdown", hovermode="x unified",
                        xaxis_title=None, yaxis_title="Filings",
                        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
                        margin=dict(l=50, r=20, t=50, b=40),
                        plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)
                    
                    cap_cols = st.columns(len(chap_cols))
                    for col_idx, col_f in enumerate(chap_cols):
                        ch_total = df_chap_fc[col_f].sum()
                        ch_peak  = df_chap_fc.loc[df_chap_fc[col_f].idxmax(), "Month"].strftime("%b %Y")
                        with cap_cols[col_idx]:
                            st.caption(f"**{col_f}**: {ch_total:.0f} projected filings (peak in {ch_peak})")
                else:
                    st.info("Chapter data unavailable.")

                st.markdown("<hr style='margin: 1.5rem 0 !important; border-top: 1px solid #cbd5e1;'>", unsafe_allow_html=True)

                df_risk_fc = run_risk_score_forecast(df_series, horizon_months=fc_horizon)
                hist_risk  = df_series.dropna(subset=["Avg_Risk"])
                fig_r = go.Figure()
                if not hist_risk.empty:
                    fig_r.add_trace(go.Scatter(
                        x=hist_risk["Month"], y=hist_risk["Avg_Risk"],
                        name="Historical Risk", mode="lines",
                        line=dict(color="#1e3a8a", width=2),
                        hovertemplate="%{x|%b %Y}  Risk: %{y:.1f}<extra></extra>"
                    ))
                if not df_risk_fc.empty:
                    fig_r.add_trace(go.Scatter(
                        x=pd.concat([df_risk_fc["Month"], df_risk_fc["Month"].iloc[::-1]]),
                        y=pd.concat([df_risk_fc["Upper Bound"], df_risk_fc["Lower Bound"].iloc[::-1]]),
                        fill="toself", fillcolor="rgba(220,38,38,0.07)",
                        line=dict(color="rgba(0,0,0,0)"), name="Conf. Band", hoverinfo="skip"
                    ))
                    fig_r.add_trace(go.Scatter(
                        x=df_risk_fc["Month"], y=df_risk_fc["Forecasted Avg Risk Score"],
                        name="Risk Forecast", mode="lines",
                        line=dict(color="#dc2626", width=2),
                        hovertemplate="%{x|%b %Y}  Forecast: %{y:.1f}<extra></extra>"
                    ))
                    fig_r.add_vline(
                        x=df_series["Month"].iloc[-1].timestamp() * 1000,
                        line_dash="dot", line_color="#94a3b8"
                    )
                fig_r.add_hline(y=90, line_dash="dot", line_color="#dc2626",
                                annotation_text="Critical (90)", annotation_position="right")
                fig_r.add_hline(y=75, line_dash="dot", line_color="#d97706",
                                annotation_text="Elevated (75)", annotation_position="right")
                fig_r.update_layout(
                    height=320, title=" Risk Score Trajectory", hovermode="x unified",
                    xaxis_title=None, yaxis_title="Avg Risk Score",
                    legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
                    margin=dict(l=50, r=70, t=50, b=40),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", range=[0, 105]),
                )
                st.plotly_chart(fig_r, use_container_width=True)
                if not df_risk_fc.empty and not hist_risk.empty:
                    latest_r = hist_risk["Avg_Risk"].iloc[-1]
                    future_r = df_risk_fc["Forecasted Avg Risk Score"].iloc[-1]
                    delta_r  = future_r - latest_r
                    direction = "⬆" if delta_r > 0 else "⬇"
                    st.caption(f"Risk at horizon end: **{future_r:.1f}** ({direction}{abs(delta_r):.1f} pts from {latest_r:.1f})")

    # -----------------------------------------------------------------
    # WORKSPACE: QUERY BOX
    # -----------------------------------------------------------------
    elif selected_tab == "Query Box":
        render_query_box_tab(schema, st.session_state.actual_schema)



if __name__ == "__main__":
    try:
        logger.info("Starting Bankruptcy application")
        main()
    except Exception as e:
        logger.exception("Fatal error in main application: %s", e)
        raise   