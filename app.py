import streamlit as st
import time
import json
import sqlite3
import pandas as pd
import sys
import importlib
import config
import logger_config

# Force reload vectorless_rag to prevent Streamlit hot-reload module caching
if "vectorless_rag" in sys.modules:
    importlib.reload(sys.modules["vectorless_rag"])

from vectorless_rag import VectorlessRAGQA, DB_PATH

logger = logger_config.setup_logger("chatbot_ui")

# ---------------------------------------------------------
# Page Configuration & Clean Elegant Light Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy QA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Elegant Styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Clean Main Header */
    .main-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .main-header h1 {
        color: #0f172a;
        font-size: 1.8rem;
        margin: 0 0 4px 0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Chat bubbles */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.01);
        margin-bottom: 12px;
    }
    
    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "followup_trigger" not in st.session_state:
    st.session_state.followup_trigger = None

def get_qa_engine():
    return VectorlessRAGQA()

qa_engine = get_qa_engine()

# ---------------------------------------------------------
# Table & Chart Rendering Utilities
# ---------------------------------------------------------
def to_bold_unicode(text):
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
    if df is not None and not df.empty:
        df = df.copy()
        df.columns = [to_bold_unicode(str(col).upper()) for col in df.columns]
    return df

def render_centered_table(df, user_query=None):
    if df is None or df.empty:
        return
    
    # Calculate percentage differences for comparison queries
    if user_query:
        q_lower = user_query.lower()
        is_comparison = any(kw in q_lower for kw in ["compare", "vs", "versus", "difference"])
        if is_comparison:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(df) >= 2 and numeric_cols:
                val_col = None
                value_keywords = ['count', 'total', 'frequency', 'value', 'sum', 'amount', 'pct', 'ratio', 'percentage']
                for col in numeric_cols:
                    if any(kw in str(col).lower() for kw in value_keywords):
                        val_col = col
                        break
                if not val_col:
                    val_col = numeric_cols[-1]
                
                baseline = float(df.iloc[0][val_col])
                if baseline != 0:
                    df = df.copy()
                    pct_changes = []
                    for idx, row in df.iterrows():
                        if idx == 0:
                            pct_changes.append("Baseline")
                        else:
                            val = float(row[val_col])
                            pct = ((val - baseline) / baseline) * 100
                            if pct > 0:
                                pct_changes.append(f'<span style="color:#16a34a; font-weight:600;">+{pct:.1f}% ▲</span>')
                            elif pct < 0:
                                pct_changes.append(f'<span style="color:#dc2626; font-weight:600;">{pct:.1f}% ▼</span>')
                            else:
                                pct_changes.append("0.0%")
                    df["Percentage Change"] = pct_changes

    html_table = format_table(df).to_html(index=False, border=0, classes="centered-results-table-el", escape=False)
    st.markdown(f"""
<style>
.centered-results-table {{
    width: 100%;
    max-height: 400px;
    overflow: auto;
    border: none;
    margin-top: 12px;
    margin-bottom: 1.5rem;
    background-color: transparent;
}}
.centered-results-table table.centered-results-table-el {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    color: #334155;
}}
.centered-results-table table.centered-results-table-el th {{
    position: sticky;
    top: 0;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 16px !important;
    text-align: center !important;
    border-bottom: 2px solid #cbd5e1 !important;
    z-index: 10;
}}
.centered-results-table table.centered-results-table-el td {{
    padding: 10px 16px !important;
    text-align: center !important;
    border-bottom: 1px solid #f1f5f9 !important;
    white-space: nowrap !important;
    transition: background-color 0.2s ease;
}}
.centered-results-table table.centered-results-table-el tr:nth-child(even) {{
    background-color: #f8fafc !important;
}}
.centered-results-table table.centered-results-table-el tr:hover td {{
    background-color: #eff6ff !important;
    color: #1e40af !important;
}}
</style>
<div class="centered-results-table">
    {html_table}
</div>
""", unsafe_allow_html=True)

def try_render_charts(df, user_query):
    """
    Analyzes the DataFrame and renders the most appropriate chart using Plotly.
    Returns True if a chart was rendered.
    """
    try:
        if df is None or df.empty or len(df.columns) < 2:
            return False
            
        # Check column types
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) != 1:
            return False
            
        num_col = numeric_cols[0]
        cat_col = [c for c in df.columns if c != num_col][0]
        
        # Case 1: Time series -> Render Line Chart
        if str(cat_col).lower() in ["year", "month", "date", "date_filed", "year_filed"]:
            import plotly.express as px
            # Sort chronologically
            df_sorted = df.copy()
            df_sorted[cat_col] = df_sorted[cat_col].astype(str)
            df_sorted = df_sorted.sort_values(by=cat_col)
            
            fig = px.line(
                df_sorted,
                x=cat_col,
                y=num_col,
                markers=True,
                color_discrete_sequence=["#1e3a8a"] # Modern dark blue
            )
            fig.update_layout(
                margin=dict(t=15, b=10, l=10, r=10),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title=None),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0", title=None)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            return True
            
        # Case 2: Categorical Distribution (<= 12 classes) -> Render Pie/Donut Chart
        if len(df) <= 12:
            import plotly.express as px
            fig = px.pie(
                df, 
                names=cat_col, 
                values=num_col, 
                hole=0.4, # Donut chart
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(
                margin=dict(t=15, b=10, l=10, r=10),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5
                )
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                hoverinfo="label+value+percent"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            return True
            
        # Case 3: Categorical Distribution (> 12 classes) -> Render Horizontal Bar Chart
        if len(df) > 12:
            import plotly.express as px
            df_sorted = df.sort_values(by=num_col, ascending=True)
            fig = px.bar(
                df_sorted,
                y=cat_col,
                x=num_col,
                orientation='h',
                color_discrete_sequence=["#2563eb"]
            )
            fig.update_layout(
                margin=dict(t=15, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title=None),
                yaxis=dict(showgrid=False, title=None)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            return True
            
    except Exception as e:
        logger.error(f"Error rendering chart: {e}")
        
    return False

def render_datasets_ui(raw_records, user_query):
    if not raw_records:
        return
        
    from insights_generator import generate_insights
    
    # 1. Multi-dataset check (planned queries plan)
    if isinstance(raw_records, list) and len(raw_records) > 0 and "data" in raw_records[0]:
        for idx, item in enumerate(raw_records):
            desc = item.get("description", "Query Dataset")
            data = item.get("data", [])
            if not data:
                continue
            df = pd.DataFrame(data)
            
            with st.container(border=True):
                st.markdown(f"#### 📊 {desc}")
                if len(df) >= 2:
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    has_chart = len(df.columns) == 2 and len(numeric_cols) == 1
                    
                    if has_chart:
                        col1, col2, col3 = st.columns([1.0, 1.0, 1.2])
                        with col1:
                            render_centered_table(df, user_query=user_query)
                        with col2:
                            try_render_charts(df, user_query=user_query)
                        with col3:
                            generate_insights(df, user_query=user_query, show_summary=False)
                    else:
                        col1, col2 = st.columns([1.0, 1.2])
                        with col1:
                            render_centered_table(df, user_query=user_query)
                        with col2:
                            generate_insights(df, user_query=user_query, show_summary=False)
                else:
                    render_centered_table(df, user_query=user_query)
                
    # 2. Standard single dataset records list
    elif isinstance(raw_records, list) and len(raw_records) > 0:
        df = pd.DataFrame(raw_records)
        with st.container(border=True):
            if len(df) >= 2:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                has_chart = len(df.columns) == 2 and len(numeric_cols) == 1
                
                if has_chart:
                    col1, col2, col3 = st.columns([1.0, 1.0, 1.2])
                    with col1:
                        render_centered_table(df, user_query=user_query)
                    with col2:
                        try_render_charts(df, user_query=user_query)
                    with col3:
                        generate_insights(df, user_query=user_query, show_summary=False)
                else:
                    col1, col2 = st.columns([1.0, 1.2])
                    with col1:
                        render_centered_table(df, user_query=user_query)
                    with col2:
                        generate_insights(df, user_query=user_query, show_summary=False)
            else:
                render_centered_table(df, user_query=user_query)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    top_k = config.DEFAULT_TOP_K
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        logger.info("User clicked 'Clear Chat History' button.")
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Main Chat Window Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Bankruptcy QA Chatbot</h1>
    <p style="color: #64748b; margin: 0; font-size: 0.95rem;">Natural language question answering with elegant tabular outputs.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" or msg.get("is_llm_synthesized", True) or not msg.get("raw_records"):
            st.markdown(msg["content"])
        
        # Render visual query understanding metadata badges if available
        if msg["role"] == "assistant" and msg.get("metadata"):
            meta = msg["metadata"]
            if "understanding" in meta:
                u = meta["understanding"]
                intent = str(u.get("intent", "N/A")).replace("_", " ").title()
                cols = ", ".join(u.get("relevant_columns", [])) if u.get("relevant_columns") else "None"
                time_f = u.get("time_filter") or "None"
                search_mode = meta.get("search_mode", "Unknown")
                latency = meta.get("retrieval_time_ms", 0.0)
                
                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: -6px; margin-bottom: 10px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; font-family: system-ui, -apple-system, sans-serif;">
                        🔍 <b>Intent</b>: {intent} &nbsp;|&nbsp; 
                        🗂️ <b>Columns</b>: <code>{cols}</code> &nbsp;|&nbsp; 
                        📅 <b>Time Constraint</b>: <code>{time_f}</code> &nbsp;|&nbsp; 
                        ⚡ <b>Engine</b>: <i>{search_mode} ({latency} ms)</i>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        if msg["role"] == "assistant" and msg.get("raw_records"):
            render_datasets_ui(msg["raw_records"], user_query=msg.get("user_query"))

# If the last message is from assistant and contains followups, display them
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    followups = last_msg.get("followup_questions")
    if followups:
        st.markdown("<div style='margin-top: 1.25rem; margin-bottom: 0.5rem; color: #475569; font-weight: 600; font-size: 0.9rem;'>💡 Suggested follow-ups:</div>", unsafe_allow_html=True)
        cols = st.columns(len(followups))
        for idx, q in enumerate(followups):
            with cols[idx]:
                if st.button(q, key=f"btn_followup_{idx}", use_container_width=True):
                    st.session_state.followup_trigger = q
                    st.rerun()

prompt = st.chat_input("Type your question here...")

if st.session_state.followup_trigger:
    prompt = st.session_state.followup_trigger
    st.session_state.followup_trigger = None

if prompt:
    logger.info(f"User submitted prompt: '{prompt}'")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Get clean text-only chat history
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            response_obj = qa_engine.generate_answer(prompt, top_k=top_k, chat_history=chat_history)
            answer_text = response_obj["answer"]
            is_llm = response_obj.get("is_llm_synthesized", True)
            followups = response_obj.get("followup_questions", [])
            
            # Extract and display visual badges instantly for the new turn
            meta = response_obj.get("metadata", {})
            if "understanding" in meta:
                u = meta["understanding"]
                intent = str(u.get("intent", "N/A")).replace("_", " ").title()
                cols = ", ".join(u.get("relevant_columns", [])) if u.get("relevant_columns") else "None"
                time_f = u.get("time_filter") or "None"
                search_mode = meta.get("search_mode", "Unknown")
                latency = meta.get("retrieval_time_ms", 0.0)
                
                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: -6px; margin-bottom: 10px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; font-family: system-ui, -apple-system, sans-serif;">
                        🔍 <b>Intent</b>: {intent} &nbsp;|&nbsp; 
                        🗂️ <b>Columns</b>: <code>{cols}</code> &nbsp;|&nbsp; 
                        📅 <b>Time Constraint</b>: <code>{time_f}</code> &nbsp;|&nbsp; 
                        ⚡ <b>Engine</b>: <i>{search_mode} ({latency} ms)</i>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            if is_llm or not response_obj.get("raw_records"):
                st.markdown(answer_text)
            render_datasets_ui(response_obj.get("raw_records"), user_query=prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "raw_records": response_obj.get("raw_records"),
        "user_query": prompt,
        "is_llm_synthesized": is_llm,
        "followup_questions": followups,
        "metadata": response_obj.get("metadata")
    })
    logger.info(f"Answer rendered on UI for prompt: '{prompt}'")
    st.rerun()
