import os
import sqlite3
import logging
import sys
import json
import re
import hashlib
import time
import random
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from config import call_llm, call_llm_haiku, call_llm_haiku2, call_llm_with_cache, count_token_usage
from dotenv import load_dotenv
from insights_generator import generate_insights

load_dotenv()

logger = logging.getLogger("bankruptcy_genbi")


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

def render_centered_table(df, user_query=None):
    if df is None or df.empty:
        return
    
    # Check if a comparison column should be added
    if user_query:
        q_lower = user_query.lower()
        is_comparison = any(kw in q_lower for kw in ["compare", "vs", "versus", "difference"])
        if is_comparison:
            # Find the most appropriate numeric value/measure column
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(df) >= 2 and numeric_cols:
                val_col = None
                value_keywords = ['count', 'total', 'frequency', 'value', 'sum', 'amount', 'pct', 'ratio', 'percentage']
                # Prefer columns containing value-like keywords
                for col in numeric_cols:
                    if any(kw in str(col).lower() for kw in value_keywords):
                        val_col = col
                        break
                # Fallback to the last numeric column (usually the aggregate measure in SQL grouping queries)
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
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
    background-color: #ffffff;
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
/* Zebra striping (2-colors) */
.centered-results-table table.centered-results-table-el tr:nth-child(even) {{
    background-color: #f8fafc !important;
}}
.centered-results-table table.centered-results-table-el tr:nth-child(odd) {{
    background-color: #ffffff !important;
}}
/* Hover effect */
.centered-results-table table.centered-results-table-el tr:hover td {{
    background-color: #eff6ff !important;
    color: #1e40af !important;
}}
/* Sleek custom scrollbars */
.centered-results-table::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
.centered-results-table::-webkit-scrollbar-track {{
    background: #f8fafc;
    border-radius: 3px;
}}
.centered-results-table::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 3px;
}}
.centered-results-table::-webkit-scrollbar-thumb:hover {{
    background: #94a3b8;
}}
</style>
<div class="centered-results-table">
    {html_table}
</div>
""", unsafe_allow_html=True)



def load_schema():
    """Load default schema from schema.json"""
    try:
        with open("schema.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None


def load_sample_data_knowledge(sample_csv_path: str = "sample_data.csv", n_rows: int = 20) -> str:
    """
    Read sample_data.csv using pd.read_csv and convert it to a compact CSV string
    that can be injected into LLM prompts as a data knowledge base.

    If the CSV file is missing or empty, falls back to querying the live data.db
    for the first n_rows rows and also refreshes the CSV file on disk.

    Returns a string of up to n_rows rows in CSV format, or an empty string on failure.
    """
    try:
        # Attempt to read from CSV file
        if os.path.exists(sample_csv_path) and os.path.getsize(sample_csv_path) > 0:
            df = pd.read_csv(sample_csv_path, nrows=n_rows)
            if not df.empty:
                logger.info("Loaded sample knowledge from %s (%d rows)", sample_csv_path, len(df))
                return df.to_csv(index=False)

        # Fallback: pull from live database
        logger.warning("sample_data.csv missing or empty — loading from data.db")
        try:
            table_name = get_db_table_name()
        except Exception:
            table_name = "uploaded_data"
        conn = sqlite3.connect("data.db")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {n_rows}", conn)
        conn.close()

        if not df.empty:
            # Persist to CSV so future calls are faster
            df.to_csv(sample_csv_path, index=False)
            logger.info("Refreshed %s from data.db (%d rows)", sample_csv_path, len(df))
            return df.to_csv(index=False)

    except Exception as e:
        logger.warning("load_sample_data_knowledge failed: %s", e)

    return ""


def clean_and_validate_suggestions(data, schema):
    """Post-process and validate suggestions to ensure they strictly conform to user rules."""
    # Strict chart types/actions that should not be in textual suggestions
    chart_only_kws = ["chart", "plot", "graph", "visualize", "visualise", "pie chart", "piechart", "bar chart", "barchart", "line chart", "linechart", "donut", "histogram", "heatmap", "scatter", "draw", "display"]
    
    # Generic visual indicators for chart detection
    visual_kws = chart_only_kws + ["breakdown", "distribution", "trend", "trends", "view", "pie", "bar", "line", "area"]
    
    textual = []
    visual = []
    
    # Extract raw suggestions
    raw_textual = data.get("textual", []) if isinstance(data, dict) else []
    raw_visual = data.get("visual", []) if isinstance(data, dict) else []
    
    # 1. Process textual suggestions: remove visual words or drop them
    for q in raw_textual:
        q_str = str(q).strip()
        if not q_str or len(q_str) < 5:
            continue
        q_lower = q_str.lower()
        
        # Check if it has any strict visual/chart keyword (ignore breakdown/distribution/trend/trends for text)
        has_strict_chart_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in chart_only_kws)
        if has_strict_chart_kw:
            # Try to rewrite it by replacing the visual action verb at the start
            cleaned_q = q_str
            cleaned_q = re.sub(r"^(show|display|draw|view|plot|visualize|visualise)\b", "", cleaned_q, flags=re.IGNORECASE).strip()
            q_lower_new = cleaned_q.lower()
            if any(re.search(rf"\b{kw}\b", q_lower_new) for kw in chart_only_kws):
                continue # discard it if it still has chart keywords
            else:
                # Rewrite leading question prefix
                if cleaned_q:
                    cleaned_q = cleaned_q[0].upper() + cleaned_q[1:]
                    if not cleaned_q.startswith(("What", "How", "List", "Filter", "Find", "Identify", "Calculate", "Compare")):
                        cleaned_q = "What is the " + cleaned_q.lower()
                    if not cleaned_q.endswith("?"):
                        cleaned_q += "?"
                    textual.append(cleaned_q)
        else:
            textual.append(q_str)
            
    # 2. Process visual suggestions: ensure they contain at least one visual keyword
    for q in raw_visual:
        q_str = str(q).strip()
        if not q_str or len(q_str) < 5:
            continue
        
        # Programmatic guard: replace any 'stacked' charts with regular/grouped bar charts
        if "stacked" in q_str.lower():
            q_str = re.sub(r"\bstacked\s+bar\s+chart(s)?\b", "Bar chart", q_str, flags=re.IGNORECASE)
            q_str = re.sub(r"\bstacked\s+bar(s)?\b", "Bar", q_str, flags=re.IGNORECASE)
            q_str = re.sub(r"\bstacked\b", "Bar", q_str, flags=re.IGNORECASE)

        q_lower = q_str.lower()
        has_visual_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in visual_kws)
        if not has_visual_kw:
            # Prepend a default chart verb
            q_str = "Show " + q_str[0].lower() + q_str[1:]
            q_lower = q_str.lower()

        # Enforce bar chart for any comparison between 2 (e.g. status vs status, chapter vs chapter, X vs Y)
        is_comparison = False
        if any(kw in q_lower for kw in [" vs ", " versus ", "compare", "comparing", "comparison"]):
            is_comparison = True
        elif any(p[0] in q_lower and p[1] in q_lower for p in [
            ("active", "closed"),
            ("chapter 7", "13"),
            ("chapter 7", "chapter 13"),
            ("new", "closed"),
            ("individual", "business"),
            ("asset", "no-asset"),
            ("asset", "no asset"),
        ]):
            is_comparison = True
            
        if is_comparison:
            # If it already has "bar" or "barchart" in it, it's fine.
            if "bar" not in q_lower and "barchart" not in q_lower:
                # Replace other chart types with Bar chart
                other_charts = ["pie chart", "pie", "donut chart", "donut", "line chart", "line", "scatter plot", "scatter", "histogram", "heatmap", "area chart", "area", "chart", "graph", "plot"]
                replaced = False
                for ct in other_charts:
                    pattern = rf"\b{ct}\b"
                    if re.search(pattern, q_lower):
                        q_str = re.sub(pattern, "Bar chart", q_str, flags=re.IGNORECASE)
                        replaced = True
                        break
                if not replaced:
                    if q_str.lower().startswith("show "):
                        q_str = "Bar chart showing " + q_str[5:]
                    elif q_str.lower().startswith("compare "):
                        q_str = "Bar chart comparing " + q_str[8:]
                    else:
                        q_str = "Bar chart of " + q_str[0].lower() + q_str[1:]
                        
        visual.append(q_str)
    
    # 2b. Schema-column validation: filter out suggestions referencing non-existent columns
    # Build a set of known column names, aliases, and domain terms from the schema
    schema_cols = {col.get("name", "").lower() for col in schema.get("columns", [])} if schema else set()
    # Add common aliases the LLM might use that map to real columns
    known_terms = schema_cols | {
        'status', 'chapter', 'state', 'client', 'attorney', 'debtor', 'trustee',
        'match_code', 'matchcode', 'match score', 'filing', 'filings', 'case', 'cases',
        'year', 'month', 'date', 'active', 'closed', 'dismissed', 'discharged',
        'pending', 'converted', 'consumer', 'individual', 'business', 'corporate',
        'partnership', 'pro se', 'prose', 'risk', 'record', 'records',
        'chapter 7', 'chapter 11', 'chapter 13', 'new', 'reopened', 'update',
        'percentage', 'count', 'total', 'average', 'top', 'bottom', 'trend',
        'distribution', 'breakdown', 'comparison', 'over time', 'by year',
    }
    # Terms the LLM commonly hallucinates that do NOT exist in the schema
    hallucinated_terms = {
        'court', 'courts', 'judge', 'judges', 'district', 'districts',
        'creditor', 'creditors', 'debt amount', 'debt', 'income',
        'asset value', 'liability', 'liabilities', 'filing fee',
        'hearing', 'hearings', 'motion', 'motions', 'claim', 'claims',
        'petition', 'petitions', 'docket', 'jurisdiction',
    }
    
    def _has_hallucinated_term(text):
        """Check if text references a hallucinated/non-existent column."""
        t_lower = text.lower()
        for term in hallucinated_terms:
            if re.search(rf'\b{term}\b', t_lower):
                return True
        return False
    
    textual = [q for q in textual if not _has_hallucinated_term(q)]
    visual = [q for q in visual if not _has_hallucinated_term(q)]
        
    # Deduplicate
    textual = list(dict.fromkeys(textual))
    visual = list(dict.fromkeys(visual))
    
    # 3. Formulate fallback lists for padding, adapted dynamically to database schema
    columns_set = {col.get("name").lower() for col in schema.get("columns", [])} if schema else set()
    
    fallback_textual = []
    if "status" in columns_set:
        fallback_textual.append("What is the active vs closed case breakdown?")
        fallback_textual.append("How many active cases are registered?")
    if "state" in columns_set:
        fallback_textual.append("Identify the top 5 states by case volume")
    if "client_name" in columns_set:
        fallback_textual.append("Identify the top client names by case volume")
    if "chapter" in columns_set:
        fallback_textual.append("What is the distribution of bankruptcy chapters?")
    if "match_code" in columns_set:
        fallback_textual.append("Compare counts of match codes")
        
    # Standard generic text questions if still lacking
    fallback_textual.extend([
        "List the most recent 10 records in the dataset",
        "Calculate the average match score for all records"
    ])
    
    fallback_textual = [f for f in fallback_textual if not any(re.search(rf"\b{kw}\b", f.lower()) for kw in visual_kws)]
    
    fallback_visual = []
    if "chapter" in columns_set:
        fallback_visual.append("Pie chart of chapter breakdown")
    if "state" in columns_set:
        fallback_visual.append("Bar chart of cases by state")
    if "date_filed" in columns_set or "open_date" in columns_set:
        fallback_visual.append("Line chart of filings over time")
    if "attorney_last_name" in columns_set:
        fallback_visual.append("Horizontal bar chart of top 10 attorneys")
    if "status" in columns_set:
        fallback_visual.append("Donut chart of case status")
        
    # Standard generic visual suggestions
    fallback_visual.extend([
        "Bar chart of filings by state",
        "Line chart of filings over time"
    ])
    
    # Pad textual to 4
    for item in fallback_textual:
        if len(textual) >= 4:
            break
        if item not in textual:
            textual.append(item)
            
    # Pad visual to 4
    for item in fallback_visual:
        if len(visual) >= 4:
            break
        if item not in visual:
            visual.append(item)
            
    return {
        "textual": textual[:4],
        "visual": visual[:4]
    }


def _generate_dynamic_prompt_suggestions(schema, conversation_memory, last_result_df=None):
    """Generate dynamic textual and visual suggestions based on conversation history
    and the columns/data of the latest result DataFrame."""
    try:
        # Build result context if we have a recent DataFrame
        result_context = ""
        if last_result_df is not None and not last_result_df.empty:
            num_cols = last_result_df.select_dtypes(include='number').columns.tolist()
            cat_cols = [c for c in last_result_df.columns if c not in num_cols]
            sample_rows = last_result_df.head(3).to_dict('records')
            result_context = (
                f"\nLAST QUERY RESULT SUMMARY:\n"
                f"- Rows returned: {len(last_result_df)}\n"
                f"- Numeric columns: {num_cols or 'none'}\n"
                f"- Categorical columns: {cat_cols or 'none'}\n"
                f"- Sample data: {sample_rows}\n"
            )

        # Format memory
        history_lines = []
        last_question_context = ""
        if conversation_memory and conversation_memory.get("history"):
            history = conversation_memory["history"]
            recent = history[-2:]
            for entry in recent:
                history_lines.append(f"User: {entry.get('user_question')}")
                if entry.get('sql_query'):
                    history_lines.append(f"SQL: {entry.get('sql_query')}")
                if entry.get('record_count') is not None:
                    history_lines.append(f"Result count: {entry.get('record_count')} rows")
            
            # Extract previous question for context
            last_q = history[-1].get("user_question")
            if last_q:
                last_question_context = (
                    f"CURRENT USER QUERY / LAST USER QUESTION: \"{last_q}\"\n"
                    f"CRITICAL REQUIREMENT:\n"
                    f"The textual suggestions MUST be a few logical follow-up questions based on the current query ('{last_q}'). "
                    f"For example, if the current query is about states and chapters, follow up on those specific states or chapters "
                    f"(e.g., 'What is the percentage of Chapter 7 in the top state?' or 'Compare Chapter 13 cases across those states'). "
                    f"Make the suggested questions feel like a natural continuation of the user's analytical journey, drilling deeper "
                    f"into the categories, filters, or time periods from the current query and its results.\n\n"
                )
        
        if history_lines:
            history_str = "Recent Conversation:\n" + "\n".join(history_lines)
        else:
            history_str = "Recent Conversation:\n(No queries asked yet. This is the start of the user's journey.)"

        # Retrieve user preferences from LangGraph InMemoryStore
        preferences_text = ""
        if "memory_store" in st.session_state:
            store = st.session_state.memory_store
            pref_item = store.get(("default-user", "preferences"), "rules")
            if pref_item and "rules" in pref_item.value:
                preferences_text = "USER PREFERENCES (Follow these rules):\n" + "\n".join([f"- {r}" for r in pref_item.value["rules"]]) + "\n\n"

        # Determine if this is the initial interaction or we hit max follow-up level (3)
        history_len = len(conversation_memory.get("history", [])) if conversation_memory else 0
        is_initial = (history_len == 0) or (history_len >= 3)

        if is_initial:
            objective_section = (
                "OBJECTIVE:\n"
                "Generate fresh, broad, exploratory business-focused questions and visualization suggestions "
                "to help the user discover the dataset as a whole.\n"
                "Do NOT generate follow-up questions. All questions must be fresh, standalone, and exploratory, treating this as a new starting point.\n\n"
            )
            relevancy_section = (
                "CRITICAL RELEVANCY RULES:\n"
                "1. The suggestions must be general and cover the entire dataset using the database schema.\n"
                "2. Do NOT reference a specific 'current client', 'filtered dataset', or any specific client names/filters from recent conversation history.\n"
                "3. Do NOT assume any specific subset or client context (e.g. do NOT say 'for the current client' or 'for active cases' unless it is a general question about all active cases in the database).\n\n"
            )
            textual_section = (
                "A. TEXTUAL QUESTIONS\n"
                "- Generate EXACTLY 4 questions.\n"
                "- Strictly give a maximum of 10 words each.\n"
                "- MUST be fresh, exploratory standalone questions about the database as a whole.\n"
                "- Business-focused and insight-driven.\n"
                "- Questions must request data, metrics, comparisons, rankings, trends, summaries, "
                "or aggregations.\n"
                "- MUST be answerable using the available schema.\n"
                "- Do NOT request charts or visualizations.\n"
                "- Avoid these words:\n"
                "  show, display, draw, chart, graph, plot, visualize, dashboard, view.\n"
                "- Prefer formats such as:\n"
                "  What is...\n"
                "  How many...\n"
                "  Which...\n"
                "  Identify...\n"
                "  Calculate...\n"
                "  Compare...\n"
                "  List...\n"
                "  Find...\n\n"
            )
            visual_section = (
                "B. VISUALIZATION SUGGESTIONS\n"
                "- Generate EXACTLY 4 suggestions.\n"
                "- Strictly give a maximum of 8 words each.\n"
                "- Explicitly mention ONE supported chart type. The ONLY supported chart types are: Bar Chart, Horizontal Bar Chart, Pie Chart, Donut Chart, Line Chart, Area Chart, Scatter Plot, Histogram, Heatmap.\n"
                "- CRITICAL: DO NOT suggest 'stacked bar chart' or 'stacked barchart'. For multi-category comparison, suggest a regular Bar Chart or Line Chart.\n"
                "- Must be business-relevant and actionable.\n"
                "- Must use actual columns or categories from the schema.\n"
                "- Prioritize trend analysis, distributions, rankings, comparisons, correlations, and geographic insights.\n"
                "- Do NOT mention 'for the current client' or use other client-specific filters.\n"
                "- For any suggestion comparing two categories, statuses, values, or items (e.g., comparing active vs closed, Chapter 7 vs 13, status vs year, or any comparison between 2), you MUST suggest a Bar Chart or Horizontal Bar Chart. Never suggest a pie chart, donut chart, line chart, or any other chart type for comparisons between 2.\n"
                "- Use phrases such as:\n"
                "  Bar chart of...\n"
                "  Line chart showing...\n"
                "  Donut chart for...\n"
                "  Heatmap of...\n"
                "  Scatter plot comparing...\n\n"
            )
            context_section = (
                "CONTEXT AWARENESS:\n"
                "- Suggest high-level exploratory questions and charts to start a fresh analytical journey.\n"
                "- Focus on uncovering patterns, concentration risks, regional trends, filing behavior, "
                "attorney performance, chapter distribution, client analysis, "
                "and temporal changes across the entire dataset.\n"
                "- CRITICAL: ONLY reference columns, categories, and values that actually exist in the DATABASE SCHEMA above. "
                "Do NOT invent or hallucinate columns like 'court', 'judge', 'district', 'creditor', 'debt amount', 'income', or 'claim'. "
                "If a concept does not map to any column in the schema, do NOT suggest it.\n\n"
            )
        else:
            objective_section = (
                "OBJECTIVE:\n"
                "Generate intelligent, business-focused follow-up questions and visualization suggestions "
                "based on:\n"
                "1. Database schema\n"
                "2. Current query result dataset\n"
                "3. Conversation history\n"
                "4. Previously explored insights\n\n"
            )
            relevancy_section = (
                "CRITICAL RELEVANCY RULES (PREVENT IRRELEVANT SUGGESTIONS):\n"
                "1. The suggestions MUST strictly build upon the active analytical context of the CURRENT USER QUERY and the columns/values shown in the LAST QUERY RESULT SUMMARY (if present).\n"
                "2. If the current query filtered by a specific value (e.g. a specific state 'NY', client 'VP', chapter '7', status 'Active', match_code 'P2'), all suggestions MUST inherit and apply that exact filter. They must ONLY ask about that specific subset (e.g. 'What is the chapter distribution in NY?', NOT 'What is the chapter distribution in CA?' or 'Compare filings by state'). Do NOT suggest questions about other values of that filtered attribute.\n"
                "3. If the previous query grouped/aggregated data (e.g., filings by state), suggest drilling down into the top group(s) from that result (e.g. 'What is the status breakdown for NY?').\n"
                "4. Do NOT suggest generic or general questions (like total volume or global state rankings) when the active query represents a narrow filtered subset.\n"
                "5. CRITICAL: Do NOT repeat the same TYPE of follow-up questions over and over. If the user already asked about a certain topic, column, or metric in the conversation history, shift focus to new dimensions (e.g., if they asked about chapters, suggest looking at statuses, timelines, or top clients next).\n\n"
            )
            textual_section = (
                "A. TEXTUAL QUESTIONS\n"
                "- Generate EXACTLY 4 questions.\n"
                "- Make the questions EXTREMELY SIMPLE and short (e.g., 'Top states by filings', 'Chapter breakdown', 'Show active cases').\n"
                "- Strictly give a maximum of 5 to 7 words each.\n"
                "- MUST be direct follow-up questions based on the current query/last user question and current query results.\n"
                "- DIVERSIFY YOUR SUGGESTIONS: Provide a mix of different analytical angles.\n"
                "- Business-focused and insight-driven.\n"
                "- Questions must request data, metrics, comparisons, rankings, trends, summaries, "
                "aggregations, filters, or anomalies that follow up logically on the previous query.\n"
                "- strictly Avoid simple questions like \"What is the total number of bankruptcy cases in the database?\".\n"
                "- MUST be answerable using the available schema and current dataset.\n"
                "- MUST reference actual columns, categories, or values from the last query results whenever possible.\n"
                "- CRITICAL: Do NOT repeat previously asked questions or the exact same type of questions asked in the conversation history.\n"
                "- Do NOT request charts or visualizations.\n"
                "- Avoid these words:\n"
                "  show, display, draw, chart, graph, plot, visualize, dashboard, view.\n"
                "- Prefer formats such as:\n"
                "  What is...\n"
                "  How many...\n"
                "  Which...\n"
                "  Identify...\n"
                "  Calculate...\n"
                "  Compare...\n"
                "  List...\n"
                "  Find...\n\n"
            )
            visual_section = (
                "B. VISUALIZATION SUGGESTIONS\n"
                "- Generate EXACTLY 4 suggestions.\n"
                "- Strictly give a maximum of 8 words each.\n"
                "- Explicitly mention ONE supported chart type. The ONLY supported chart types are: Bar Chart, Horizontal Bar Chart, Pie Chart, Donut Chart, Line Chart, Area Chart, Scatter Plot, Histogram, Heatmap.\n"
                "- CRITICAL: DO NOT suggest 'stacked bar chart' or 'stacked barchart'. For multi-category comparison, suggest a regular Bar Chart or Line Chart.\n"
                "- Must be business-relevant and actionable.\n"
                "- Must use actual columns, categories, or values from the dataset.\n"
                "- Prioritize trend analysis, distributions, rankings, comparisons, correlations, and geographic insights.\n"
                "- For any suggestion comparing two categories, statuses, values, or items (e.g., comparing active vs closed, Chapter 7 vs 13, status vs year, or any comparison between 2), you MUST suggest a Bar Chart or Horizontal Bar Chart. Never suggest a pie chart, donut chart, line chart, or any other chart type for comparisons between 2.\n"
                "- Use phrases such as:\n"
                "  Bar chart of...\n"
                "  Line chart showing...\n"
                "  Donut chart for...\n"
                "  Heatmap of...\n"
                "  Scatter plot comparing...\n\n"
            )
            context_section = (
                "CONTEXT AWARENESS:\n"
                "- If prior results exist, generate deeper investigative follow-ups applying the relevancy rules.\n"
                "- Use actual values from the latest result summary whenever available.\n"
                "- Focus on uncovering patterns, concentration risks, regional trends, filing behavior, "
                "attorney performance, chapter distribution, client analysis, "
                "and temporal changes.\n"
                "- CRITICAL: ONLY reference columns, categories, and values that actually exist in the DATABASE SCHEMA above. "
                "Do NOT invent or hallucinate columns like 'court', 'judge', 'district', 'creditor', 'debt amount', 'income', or 'claim'. "
                "If a concept does not map to any column in the schema, do NOT suggest it.\n\n"
            )

        prompt = (
            "You are an expert AI analyst for a Bankruptcy Analytics Dashboard.\n\n"
            f"{objective_section}"
            f"DATABASE SCHEMA:\n{json.dumps(schema, indent=2)[:200000]}\n\n"
            f"{history_str}\n"
            f"{preferences_text}"
            f"{result_context}\n\n"
            f"{last_question_context}"
            "SUPPORTED VISUALIZATION TYPES:\n"
            "- Bar Chart\n"
            "- Horizontal Bar Chart\n"
            "- Pie Chart\n"
            "- Donut Chart\n"
            "- Line Chart\n"
            "- Area Chart\n"
            "- Scatter Plot\n"
            "- Histogram\n"
            "- Heatmap\n\n"
            f"{relevancy_section}"
            "QUESTION GENERATION RULES:\n\n"
            f"{textual_section}"
            f"{visual_section}"
            f"{context_section}"
            "OUTPUT FORMAT:\n"
            "Return ONLY valid JSON.\n"
            "Do NOT return markdown, explanations, notes, comments, or additional text.\n\n"
            "{\n"
            '  "textual": [\n'
            '    "Question 1",\n'
            '    "Question 2",\n'
            '    "Question 3",\n'
            '    "Question 4"\n'
            "  ],\n"
            '  "visual": [\n'
            '    "Visual 1",\n'
            '    "Visual 2",\n'
            '    "Visual 3",\n'
            '    "Visual 4"\n'
            "  ]\n"
            "}"
        )



        response = call_llm_haiku2(prompt)
        
        logger.info("Generated suggestions via Haiku2: %s", response)

        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\n", "", cleaned_response)
                cleaned_response = re.sub(r"\n```$", "", cleaned_response)

            data = json.loads(cleaned_response)
            if isinstance(data, dict) and "textual" in data and "visual" in data:
                return clean_and_validate_suggestions(data, schema)
        except Exception as ex:
            logger.warning("Failed to parse Haiku JSON response: %s. Using regex fallback.", ex)

        # Fallback regex parsing
        textual, visual = [], []
        matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)
        for m in matches:
            if m not in ["textual", "visual"] and len(m) > 8:
                m_lower = m.lower()
                if any(kw in m_lower for kw in ["chart", "plot", "graph", "visualize", "bar", "pie", "line", "donut", "scatter", "histogram", "heatmap", "area"]):
                    if len(visual) < 4:
                        visual.append(m)
                else:
                    if len(textual) < 4:
                        textual.append(m)

        if len(textual) >= 1 or len(visual) >= 1:
            return clean_and_validate_suggestions({"textual": textual, "visual": visual}, schema)

    except Exception as e:
        logger.exception("Error generating dynamic suggestions: %s", e)

    # Fallback suggestions generated dynamically from the schema if the LLM call fails
    columns_set = {col.get("name").lower() for col in schema.get("columns", [])} if schema else set()
    fallback_textual = []
    fallback_visual = []

    if "state" in columns_set:
        fallback_textual.append("Top 10 states with the most filings")
        fallback_visual.append("Bar chart of filings by state")
    if "status" in columns_set:
        fallback_textual.append("What is active vs closed case breakdown?")
    if "chapter" in columns_set:
        fallback_visual.append("Pie chart of chapter distribution")
    if "open_date" in columns_set or "date_filed" in columns_set:
        fallback_visual.append("Line chart of filings by year")
    if "attorney_last_name" in columns_set or "attorney_first_name" in columns_set or "attorney_dba" in columns_set:
        fallback_visual.append("Horizontal bar chart of top 10 attorneys")

    # Fill in generic ones to meet target counts
    if len(fallback_textual) < 4:
        fallback_textual.append("How many total bankruptcy filings are there?")
    if len(fallback_textual) < 4:
        fallback_textual.append("List the most recent 10 records")
    if len(fallback_textual) < 4:
        fallback_textual.append("Filter cases by Chapter 7")
    if len(fallback_textual) < 4:
        fallback_textual.append("Identify the top 10 states by volume")

    if len(fallback_visual) < 4:
        fallback_visual.append("Pie chart of chapter distribution")
    if len(fallback_visual) < 4:
        fallback_visual.append("Bar chart of filings by state")
    if len(fallback_visual) < 4:
        fallback_visual.append("Line chart of filings by year")
    if len(fallback_visual) < 4:
        fallback_visual.append("Horizontal bar chart of top 10 states")

    return clean_and_validate_suggestions({"textual": fallback_textual, "visual": fallback_visual}, schema)


def get_db_table_name():
    """Detect active table name in SQLite database"""
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='uploaded_data' OR name='bankruptcy4')")
        res = cursor.fetchall()
        conn.close()
        if res:
            names = [r[0] for r in res]
            if "uploaded_data" in names:
                return "uploaded_data"
            return names[0]
        return "uploaded_data"
    except Exception:
        return "uploaded_data"


def get_actual_database_schema():
    """Dynamically construct schema based on the current active table in data.db"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        conn.close()
        
        schema = {
            "table_name": table_name,
            "columns": []
        }
        
        descriptions = {}
        try:
            with open("schema.json", "r") as f:
                static_schema = json.load(f)
                descriptions = {col["name"].lower(): col.get("description", "") for col in static_schema.get("columns", [])}
        except Exception:
            pass
            
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            clean_name = clean_column_name(col_name)
            schema["columns"].append({
                "name": col_name,
                "type": col_type,
                "description": descriptions.get(clean_name, f"Database field: {col_name}")
            })
        return schema
    except Exception as e:
        logger.error(f"Error getting dynamic database schema: {e}")
        return None


# =============================================================================
# SMART QUERY UNDERSTANDING (HAIKU PRE-PROCESSOR)
# =============================================================================

def smart_query_understanding(user_query: str, schema: dict, conversation_memory: dict = None) -> dict:
    """
    Use a dedicated Claude Haiku session to understand any user question by
    referencing the full schema.json as a knowledge base.

    Returns a dict with:
      - normalized_query: A rewritten, schema-aware version of the user query
      - intent: One of 'data_retrieval', 'aggregation', 'filter', 'visualization', 'follow_up', 'unclear'
      - relevant_columns: List of column names from the schema that are relevant
      - time_filter: Any detected time/date constraint (or None)
      - is_answerable: True if the query can be answered from the schema
      - clarification_needed: A short message if the query is unclear (or None)
    """
    try:
        # Build a compact schema summary for the prompt
        schema_cols = schema.get("columns", []) if schema else []
        table_name = schema.get("table_name", "bankruptcy_data") if schema else "bankruptcy_data"
        schema_summary_lines = [
            f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}"
            for col in schema_cols
        ]
        schema_text = f"Table: {table_name}\nColumns:\n" + "\n".join(schema_summary_lines)

        # Build recent conversation context
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-2:]
            lines = []
            for entry in recent:
                lines.append(f"  User: {entry.get('user_question', '')}")
                if entry.get("sql_query"):
                    lines.append(f"  SQL: {entry.get('sql_query')}")
                    lines.append(f"  SQL result: {entry.get('record_count', 0)} rows")
            conv_context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
        # Retrieve user preferences from LangGraph InMemoryStore
        preferences_text = ""
        if "memory_store" in st.session_state:
            store = st.session_state.memory_store
            pref_item = store.get(("default-user", "preferences"), "rules")
            if pref_item and "rules" in pref_item.value:
                preferences_text = "USER PREFERENCES (Follow these rules):\n" + "\n".join([f"- {r}" for r in pref_item.value["rules"]]) + "\n\n"

        prompt = (
            "You are a smart query understanding assistant for a bankruptcy case management dashboard.\n"
            "Your job is to analyze the user's question and map it to the database schema below.\n\n"
            f"DATABASE SCHEMA:\n{schema_text}\n\n"
            f"{conv_context}"
            f"{preferences_text}"
            "USER QUESTION: \"" + user_query + "\"\n\n"
            "KEY COLUMN MAPPING RULES (use these to resolve ambiguity):\n"
            "- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' → use the 'status' column (values: Active, Closed, Dismissed, Converted, Pending, Discharged). Note: if the query specifically requests a comparison/breakdown between specific statuses (like 'active vs closed') along with another dimension (e.g. by year, by state), make sure the normalized query explicitly asks for both grouping dimensions (e.g. 'Show count of cases grouped by status AND state/year').\n"
            "- 'active/inactive flag' → use the 'active_status' column\n"
            "- 'new/closed/reopened/update record stage' → use the 'record_type' column (values: New, Closed, Reopened, Update)\n"
            "- CRITICAL DISAMBIGUATION: 'New' is EXCLUSIVELY a record_type value (NEVER a status value). When the user says 'new vs closed', 'closed vs new', 'new cases', always use 'record_type' column. Use 'status' only for legal status values like Active, Pending, Dismissed, Discharged, Converted.\n"
            "- 'bankruptcy chapter', 'chapter 7/11/13' → use the 'chapter' column\n"
            "- 'filing date', 'when filed' → use the 'date_filed' column\n"
            "- 'debtor name', 'customer name' → use 'first_name' and 'last_name' (perform a partial match LIKE search)\n"
            "- 'state' (debtor location) → use the 'state' column\n"
            "- 'client', 'client name', 'client code', 'VP', 'SYF', 'carecredit' or any client identifier → use the 'client_name' column. NEVER map client terms to 'match_code'.\n"
            "- 'attorney', 'lawyer' → use the 'attorney_first_name', 'attorney_last_name' or 'attorney_dba' columns. When analyzing or grouping by attorney, use 'attorney_dba' (or include first and last names along with 'attorney_dba') to represent different practitioners/firms.\n"
            "- 'year', 'yearly', 'by year', 'annual' → use the 'date_filed' or 'open_date' column (or date columns) and extract the year using SQLite's strftime('%Y', ...) function.\n"
            "- 'matchcode', 'match_code' (values like P1, P2, P3, M1, M2, M3) → use the 'match_code' column.\n"
            "- 'P2 matchcode' → matches the 'match_code' column with value 'P2'.\n"
            "- 'P3 and M3' or other multiple match codes → use the 'match_code' column with multiple matching values ('P3' or 'M3').\n"
            "- 'partnership cases' → use 'consumer_type' = 'Partnership'\n"
            "- 'business cases' → use 'consumer_type' = 'Business'\n"
            "- 'corporate' → use 'consumer_type' = 'Corporate'\n"
            "- 'prose', 'pro se', 'self-represented' → use 'prose_indicator' = 'Y'\n"
            "- 'without attorney', 'no attorney' → check if 'prose_indicator' = 'Y' OR attorney columns (like attorney_first_name) are NULL or empty\n"
            "- 'held cases' or 'cases held' → use 'status' = 'Pending'\n"
            "- 'transfers' in corporate context → use 'status' = 'Converted' or check 'conversion_date' column\n"
            "- 'high risk' cases → use 'match_score' >= 98 (since database match scores range from 80 to 99)\n"
            "- 'individual breakdown for this' or other breakdown follow-up questions → Look at the previous query's SQL. If the previous query filtered by a column (e.g. TRIM(match_code) IN ('P3', 'M3')), rewrite the follow-up to ask for counts grouped by that column (e.g. 'Show count of cases grouped by match_code column where match_code is P3 or M3'). Inherit the filters of the previous query.\n"
            "- Pronoun follow-ups like 'who are they', 'who are those', 'what are they', 'list them' asking for the entities behind a count or aggregation → Rewrite to list/retrieve the distinct names or values of that entity (e.g. 'Show the unique client_name values' or 'List the unique client_names'). Do NOT count them again. Avoid 'count' or 'how many' in the normalized query when the user is asking 'who' or 'what' to list the actual entities.\n\n"
            "INSTRUCTIONS:\n"
            "1. Understand the user's intent (data_retrieval, aggregation, filter, visualization, follow_up, or unclear).\n"
            "2. Rewrite the question as a clear, unambiguous PLAIN ENGLISH query using EXACT column names from the schema.\n"
            "   IMPORTANT: normalized_query must be a natural language sentence, NOT SQL code.\n"
            "   - Map informal terms to schema columns using the KEY COLUMN MAPPING RULES above.\n"
            "   - Use ONLY column names that exist in the schema. Never invent column names.\n"
            "   - Expand abbreviations and correct spelling based on schema knowledge.\n"
            "   - Preserve all specific filter values (e.g. state names like 'NY', status names like 'Active', chapter numbers like '7', and match codes like 'P2' or 'M1'). Never generalize specific filter values or drop them.\n"
            "   - Keep limit, ordering, and superlative constraints explicitly in the normalized query. For example:\n"
            "     * For singular superlative questions (e.g., 'Which state has the most filings', 'Which chapter has the least cases', 'Which year has the highest filings', 'Which attorney has the highest percent of active cases'), rewrite it to include 'showing only the top 1' or 'showing only the bottom 1'.\n"
            "     * For plural superlative questions (e.g., 'Which states have the highest number of filings', 'top states by cases', 'Which attorneys have the highest number of active cases'), rewrite it to include 'showing only the top 5' (or the limit specified, or 'showing only the top 10'). Do not default to top 1 for plural questions.\n"
            "     * For general distribution, grouping, or yearly/monthly breakdown questions without superlatives (e.g., 'How many filings each year', 'filings by state', 'breakdown by chapter', 'cases received each year'), do NOT include any limit restriction like 'showing only the top 1' or 'showing only the top 5'. These questions should return the complete breakdown.\n"
            "     * For dual superlative questions asking for BOTH extremes (e.g., 'Which years had the highest and lowest number of cases', 'Which states had the most and least filings', 'top and bottom chapters'), rewrite the question to ask for the count grouped by that category showing only the top 1 and bottom 1 combined.\n"
            "   - If the user asked for a 'bar chart', 'pie chart', 'plot', etc., keep those words in the normalized_query.\n"
            "   - Example: 'who are they?' (when previous query was 'How many unique clients') → 'Retrieve the unique client_name values'\n"
            "   - Example: 'what are those?' (when previous query was 'How many match codes are there') → 'Retrieve the unique match_code values'\n"
            "   - Example: 'Which state has the highest number of filings?' → 'Show count of cases grouped by state column ordered by count descending showing only the top 1'\n"
            "   - Example: 'Which year had the highest number of filings?' → 'Show count of cases grouped by year (extracted from date_filed column) ordered by count descending showing only the top 1'\n"
            "   - Example: 'Which years had the highest and lowest number of cases?' → 'Show count of cases grouped by year (extracted from date_filed column) showing only the top 1 and bottom 1 combined'\n"
            "   - Example: 'How many new bankruptcy filings are received each year?' → 'Show count of cases grouped by year (extracted from date_filed column) ordered by year'\n"
            "   - Example: 'Which states have the highest number of active filings?' → 'Show count of cases grouped by state column ordered by count descending showing only the top 5 where status is Active'\n"
            "   - Example: 'Which chapter has the least cases?' → 'Show count of cases grouped by chapter column ordered by count ascending showing only the top 1'\n"
            "   - Example: 'top 3 states with most filings' → 'Show count of cases grouped by state column ordered by count descending showing only the top 3'\n"
            "   - Example: 'bar chart of filings by state' → 'Show a bar chart of the count grouped by state column'\n"
            "   - Example: 'show debtors in NY' → 'Retrieve records where state = NY showing first_name, last_name, city, state'\n"
            "   - Example: 'active vs closed case breakdown' → 'Show count of cases grouped by status column filtered to show only status is Active or Closed'\n"
            "   - Example: 'Bar chart comparing active vs closed cases by year' → 'Show a bar chart of the count of cases grouped by status and year (extracted using strftime from date_filed) filtered to show only status is Active or Closed'\n"
            "   - Example: 'Compare active vs closed by state' → 'Show count of cases grouped by status and state columns filtered to show only status is Active or Closed'\n"
            "   - Example: 'Analyze filings by year and status' → 'Retrieve counts of records grouped by the year (extracted using strftime from date_filed or open_date column) and status column'\n"
            "   - Example: 'Show chapter distribution for each state' → 'Retrieve counts of records grouped by state and chapter columns showing both columns'\n"
            "   - Example: 'P2 matchcode yearwise distribution' → 'Show count of cases grouped by year (extracted from date_filed) where match_code is P2'\n"
            "   - Example: 'VP yearwise distribution' → 'Show count of cases grouped by year (extracted from date_filed) where client_name is VP'\n"
            "   - Example: 'VP closed vs new cases' → 'Show count of cases grouped by record_type column where client_name is VP and record_type is New or Closed'\n"
            "   - Example: 'Partnership Cases without attorney' → 'Retrieve records where consumer_type is Partnership and (prose_indicator is Y or attorney_first_name is empty/null)'\n"
            "   - Example: 'ProSe cases of vp' → 'Retrieve records where client_name is VP and prose_indicator is Y'\n"
            "   - Example: 'P3 and M3 total volume' → 'Show the total count of records where match_code is P3 or M3'\n"
            "   - Example: 'Corporate transfers in NY' → 'Retrieve records where consumer_type is Corporate and state is NY and status is Converted or conversion_date is not null'\n"
            "   - Example: 'show high risk cases' → 'Retrieve records where match_score is greater than or equal to 98 ordered by match_score descending'\n"
            "   - Example: 'show top 10 high risk cases' → 'Retrieve records where match_score is greater than or equal to 98 ordered by match_score descending showing only the top 10'\n"
            "   - Example: 'show top 10 cases for TX' → 'Retrieve records where state = TX showing account_number, first_name, last_name, state, date_filed, status, match_score ordered by match_score descending showing only the top 10'\n"
            "   - Example: 'show all chapters filing counts' → 'Show count of cases grouped by chapter column ordered by count descending'\n"
            "   - Example: 'High risk cases yearwise' → 'Show count of cases grouped by year (extracted from date_filed) where match_score is greater than or equal to 98'\n"
            "   - Example: 'Top client filed chapter 7' → 'Retrieve the client_name that has the highest count of records where chapter is 7'\n"
            "   - Example: 'show chapter 13 business cases' → 'Retrieve records where chapter is 13 and consumer_type is Business'\n"
            "   - Example: 'Show held cases for 2024' → 'Retrieve records where status is Pending and year (extracted from date_filed) is 2024'\n"
            "   - Example: 'details of customer XYZ' → 'Retrieve records where first_name or last_name matches XYZ using LIKE partial match'\n"
            "   - Example (follow-up breakdown): 'can you give me individual breakdown for this?' (when previous query was P3 and M3 volume) → 'Show count of cases grouped by match_code column where match_code is P3 or M3'\n"
            "3. List the relevant column names from the schema (use exact column names).\n"
            "4. Extract any time/date filter mentioned (e.g., '2024', 'last year', 'Q1 2023') or null.\n"
            "5. Determine if the question is answerable from the schema (true/false).\n"
            "6. If unclear or unanswerable, provide a brief clarification message.\n\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "{\n"
            '  "normalized_query": "plain English rewrite - never SQL",\n'
            '  "intent": "data_retrieval|aggregation|filter|visualization|follow_up|unclear",\n'
            '  "relevant_columns": ["col1", "col2"],\n'
            '  "time_filter": "2024" or null,\n'
            '  "is_answerable": true or false,\n'
            '  "clarification_needed": "..." or null\n'
            "}\n"
            "Do NOT include markdown, SQL code, or any text outside the JSON."
        )

        response = call_llm_haiku(prompt)
        logger.info("Smart query understanding response: %s", response[:300] if response else "EMPTY")

        # Parse the JSON response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        result = json.loads(cleaned)

        # Validate: reject normalized_query if it looks like SQL
        normalized = result.get("normalized_query", user_query).strip()
        if normalized.strip().upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
            logger.warning("Haiku returned SQL as normalized_query — falling back to user_query")
            normalized = user_query

        return {
            "normalized_query": normalized,
            "intent": result.get("intent", "data_retrieval"),
            "relevant_columns": result.get("relevant_columns", []),
            "time_filter": result.get("time_filter"),
            "is_answerable": bool(result.get("is_answerable", True)),
            "clarification_needed": result.get("clarification_needed"),
        }

    except Exception as e:
        logger.warning("Smart query understanding failed (falling back to original query): %s", e)
        return {
            "normalized_query": user_query,
            "intent": "data_retrieval",
            "relevant_columns": [],
            "time_filter": None,
            "is_answerable": True,
            "clarification_needed": None,
        }


# =============================================================================
# SQL CONVERSION & VALIDATION
# =============================================================================

def extract_sql_from_response(text: str) -> str:
    """Robust parser to extract SQL query string from JSON, markdown or raw text output."""
    try:
        if not text:
            return ""
        
        # 1. Try to extract value from "sql" or "CORRECTED_QUERY" key via regex (handles malformed/escaped JSON)
        for key in ["sql", "CORRECTED_QUERY"]:
            m_field = re.search(r'"' + key + r'"\s*:\s*"([\s\S]*?)"', text, re.I)
            if m_field:
                val = m_field.group(1).strip()
                # Unescape common escaped characters in JSON strings
                val = val.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
                if val.lower().strip().startswith("select"):
                    return val

        # 2. Try to find and parse a JSON block anywhere in the text
        m_json = re.search(r"(\{[\s\S]*\})", text)
        if m_json:
            json_str = m_json.group(1)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "sql" in data:
                    return data["sql"].strip()
                elif isinstance(data, dict) and "CORRECTED_QUERY" in data:
                    return data["CORRECTED_QUERY"].strip()
            except Exception:
                try:
                    cleaned_str = json_str.replace("\\'", "'")
                    data = json.loads(cleaned_str)
                    if isinstance(data, dict) and "sql" in data:
                        return data["sql"].strip()
                    elif isinstance(data, dict) and "CORRECTED_QUERY" in data:
                        return data["CORRECTED_QUERY"].strip()
                except Exception:
                    pass

        # 3. Try to extract from ```sql ... ``` block
        m_sql_block = re.search(r"```sql\s*([\s\S]*?)```", text, re.I)
        if m_sql_block:
            return m_sql_block.group(1).strip()

        # 4. Try to extract from standard ``` ... ``` block
        m_block = re.search(r"```\s*([\s\S]*?)```", text, re.I)
        if m_block:
            content = m_block.group(1).strip()
            # If the content contains "json" at the beginning, strip it and try parsing
            if content.lower().startswith("json"):
                try:
                    cleaned_json = content[4:].strip()
                    try:
                        data = json.loads(cleaned_json)
                        if isinstance(data, dict) and "sql" in data:
                            return data["sql"].strip()
                    except Exception:
                        cleaned_json_fixed = cleaned_json.replace("\\'", "'")
                        data = json.loads(cleaned_json_fixed)
                        if isinstance(data, dict) and "sql" in data:
                            return data["sql"].strip()
                except Exception:
                    pass
            # If it's just raw SQL inside the block (and not a JSON string), return it
            if "select" in content.lower() and not (content.strip().startswith("{") or '"sql"' in content.lower()):
                return content
            
        # 5. Regex fallback for raw SELECT statement
        m_select = re.search(r"\b(SELECT[\s\S]*?);?\s*$", text, re.I)
        if m_select:
            sql = m_select.group(1).strip()
            # Clean up trailing markdown/JSON characters if the LLM output was garbled
            sql = re.sub(r'["\'\}]+$', '', sql).strip()
            return sql

        return ""
    except Exception as e:
        logger.exception("Error extracting SQL: %s", e)
        return ""


def intent_classifier(user_question: str, conversation_memory: dict = None, token_usage: dict = None) -> str:
    """Stage 1: Intent Classifier - Classify user question intent."""
    try:
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-2:]
            lines = []
            for entry in recent:
                lines.append(f"User: {entry.get('user_question', '')}")
                lines.append(f"Assistant: {entry.get('assistant_answer', '') or entry.get('sql_query', '')}")
            conv_context = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are an Intent Classification assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Your job is to analyze the user's question and conversation history to classify the user's intent.\n\n"
            f"{conv_context}"
            f"USER QUESTION: \"{user_question}\"\n\n"
            "Choose exactly one of the following classification labels:\n"
            "- 'data_retrieval': requests a list or raw rows of data (e.g., 'show latest 10 filings', 'list attorneys in NY', 'show top 3 risk cases', 'highest score cases'). Note: queries asking for 'top N', 'latest N', 'highest/lowest risk cases', 'highest match scores', or listing cases/records are 'data_retrieval' intent, NOT aggregation.\n"
            "- 'aggregation': requests counts, sums, averages, grouping, or statistics (e.g., 'total filings', 'cases by state', 'average match score').\n"
            "- 'filter': requests filtering records based on values without aggregation (e.g., 'show only active cases').\n"
            "- 'visualization': explicitly requests a chart, plot, pie, bar, graph, or distribution representation (e.g., 'draw a pie chart of chapter distribution', 'plot filings over time').\n"
            "- 'follow_up': references the previous question/results, or uses pronouns/reference words like 'these', 'those', 'that', 'convert to chart' (e.g., 'filter those by active status', 'plot that').\n"
            "- 'unclear': off-topic queries, greetings, or ambiguous text (e.g., 'hello', 'what is bankruptcy').\n\n"
            "Respond ONLY with a JSON object containing the reasoning, confidence, and intent, formatted as:\n"
            "{\n"
            '  "reasoning": "<one sentence explaining the user intent classification>",\n'
            '  "confidence": <confidence score from 0.0 to 1.0>,\n'
            '  "intent": "data_retrieval|aggregation|filter|visualization|follow_up|unclear"\n'
            "}\n"
            "Do not include markdown, code blocks, or explanation text outside the JSON."
        )
        
        intent_model = os.getenv("INTENT_DETECTION_MODEL_ID") or "anthropic.claude-3-haiku-20240307-v1:0"
        response = call_llm_with_cache(prompt, model=intent_model, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v
        
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            
        data = json.loads(cleaned)
        logger.info(
            "Intent Classifier | detected=%s | confidence=%s | reasoning=%r",
            data.get("intent"),
            data.get("confidence"),
            data.get("reasoning"),
        )
        return data.get("intent", "data_retrieval")
    except Exception as e:
        logger.warning("Intent classification failed, falling back to 'data_retrieval': %s", e)
        return "data_retrieval"


def entity_extractor(user_question: str, intent: str, conversation_memory: dict = None, token_usage: dict = None) -> dict:
    """Stage 2: Entity Extractor - Extract filter values, fields, and constraints."""
    try:
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-2:]
            lines = []
            for entry in recent:
                lines.append(f"User: {entry.get('user_question', '')}")
                if entry.get("sql_query"):
                    lines.append(f"SQL: {entry.get('sql_query')}")
            conv_context = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are an Entity Extraction assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Extract entities, filter criteria, parameters, and constraints from the user's question.\n\n"
            f"INTENT: {intent}\n"
            f"{conv_context}"
            f"USER QUESTION: \"{user_question}\"\n\n"
            "Identify and extract the following entity types if present (use null if not found):\n"
            "1. status: e.g. Active, Closed, Dismissed, Converted, Pending, Discharged. Can be a single string (e.g. 'Active') or a list (e.g. ['Active', 'Closed']) for comparisons. (Note: if user says 'held', set status to 'Pending')\n"
            "2. chapter: e.g. 7, 11, 13. Can be a single integer (e.g. 7) or a list (e.g. [11, 13]) for comparisons.\n"
            "3. state: e.g. NY, CA, TX, Florida. Can be a single string or a list for comparisons.\n"
            "4. date_or_year: e.g. 2024, last year, since 2023, open date range.\n"
            "5. attorney_name: e.g. John Doe, Smith.\n"
            "6. debtor_name: debtor first or last name, or general customer name (e.g. 'XYZ').\n"
            "7. match_code: e.g. P1, P2, P3, M1, M2, M3. Can be a single string (e.g. 'P2') or a list of strings (e.g. ['P3', 'M3']) for multi-code queries (ONLY set this for match code values — never for client names like VP, SYF, or carecredit).\n"
            "8. client_name: client or lender code, e.g. VP, SYF, SYNC, carecredit. Set this when the user mentions a client name/code like 'VP yearwise', 'carecredit cases'. NEVER confuse with match_code.\n"
            "9. aggregation_type: count, sum, average, none.\n"
            "10. group_by_fields: fields to group results by (e.g. state, chapter, year, match_code, client_name, trustee_city, record_type, status).\n"
            "11. limit: number of records to return.\n"
            "12. sort_order: asc, desc, or null.\n"
            "13. city: general city name or debtor city (e.g. Houston, Chicago).\n"
            "14. trustee_name: trustee name.\n"
            "15. trustee_city: trustee city (e.g. Albany, Boston).\n"
            "16. sort_by_field: field to sort or order the results by (e.g. risk, score, date, open date).\n"
            "17. record_type: lifecycle stage of the bankruptcy record. Values: New, Update, Reopened, Closed. Use when the user mentions 'new cases', 'new filings', 'closed records', 'reopened cases', 'new vs closed'. Can be a single string (e.g. 'New') or a list of strings for comparisons (e.g. ['New', 'Closed']).\n"
            "18. consumer_type: Type of debtor/consumer. Values: Individual, Business, Corporate, Partnership, Trust, SME. (e.g. 'Partnership cases' -> 'Partnership', 'business cases' -> 'Business', 'Corporate' -> 'Corporate')\n"
            "19. prose_indicator: Whether debtor is self-represented (pro se). 'Y' = pro se, 'N' = has attorney.\n"
            "20. has_no_attorney: Boolean flag (true/false). Set to true when user says 'without attorney', 'no attorney', 'no lawyer'.\n\n"
            "CRITICAL CONTEXT ISOLATION RULE:\n"
            "- The 'Recent Conversation History' above is provided ONLY as context to resolve pronouns (e.g. 'it', 'that', 'those', 'same state') or vague references in the current question.\n"
            "- NEVER carry forward filter values (e.g. state, chapter, status, date_or_year, client_name, match_code) from previous SQL queries or previous questions into the current extraction UNLESS the current question explicitly references them by name or pronoun.\n"
            "- If the current question explicitly names a NEW specific value (e.g. 'Texas', 'Chapter 7', 'Active'), extract ONLY that value — do NOT add or merge any values seen in previous SQL queries.\n"
            "- Example: If the previous SQL was 'WHERE state IN (''TX'', ''AZ'')' and the current question is 'Pie chart showing the distribution of bankruptcy chapters in Texas', extract state='TX' ONLY. Do NOT include 'AZ' just because it appeared in a prior SQL query.\n\n"
            "KEY PARSING RULE FOR COLLOQUIAL / INCORRECT ENGLISH:\n"
            "- Suffixes like 'wise' (e.g. yearwise, chapterwise, statewise, matchcodewise) specify fields that must be placed inside the 'group_by_fields' list (e.g. ['year'], ['chapter'], ['state'], ['match_code']).\n"
            "- Abbreviated filters like 'P2 matchcode' should set 'match_code' to 'P2'.\n"
            "- Client codes like 'VP yearwise', 'SYF distribution' should set 'client_name' to the code (e.g. 'VP') and group_by_fields to ['year']. Do NOT set match_code for client names.\n"
            "- If the user asks for a distribution or breakdown for a single year/date (e.g., 'VP 2024 distributions', '2024 cases distribution'), set 'date_or_year' to that year (e.g. '2024') and set 'group_by_fields' to ['month'] (instead of 'year') so that the distribution is shown across the months of that year.\n"
            "- Phrasings like 'NY attorney list' should set 'state' to 'NY' and keep track of other attributes.\n"
            "- Phrasings like 'trusty city -Albany' or 'trustee city Albany' should set 'trustee_city' to 'Albany'. Do NOT map city names (e.g., 'Albany') to the 'state' field as 'NY' or any other state abbreviation.\n"
            "- Phrasings like 'trustee name Smith' should set 'trustee_name' to 'Smith'.\n"
            "- Phrasings like 'top 3 risk cases' or 'highest risk cases' should set 'limit' to 3 (or the specified number), 'sort_by_field' to 'risk', and 'sort_order' to 'desc'.\n"
            "- SUPERLATIVES ON AGGREGATIONS (e.g. 'most filings', 'least cases', 'highest count', 'lowest filings'):\n"
            "- COMPARISONS: If the user asks to compare two or more specific categories (e.g. 'Chapter 11 and Chapter 13', 'Active vs Closed'), you MUST extract ALL values into the corresponding field as a list (e.g. chapter: [11, 13], status: ['Active', 'Closed']). If another grouping dimension (e.g. state, year) is requested, set BOTH columns in 'group_by_fields' (e.g. ['state', 'status'] or ['year', 'status']). Otherwise, set 'group_by_fields' to that column (e.g. ['status'] or ['chapter']).\n"
            "  * If the question asks for 'most', 'highest', 'maximum', 'top', or 'best' of a count or aggregation:\n"
            "    - If the target noun is singular (e.g., 'Which state has...', 'the top chapter'), set 'limit' to 1 (or the number specified like 'top 3' -> 3).\n"
            "    - If the target noun is plural (e.g., 'Which states have...', 'top chapters', 'highest states'), do NOT set 'limit' unless a specific number is requested (e.g., set to null, unless user says 'top 5').\n"
            "    - Set 'sort_order' to 'desc' and 'sort_by_field' to 'count'.\n"
            "  * If the question asks for 'least', 'lowest', 'minimum', 'bottom', or 'worst' of a count or aggregation:\n"
            "    - If the target noun is singular, set 'limit' to 1 (or the number specified).\n"
            "    - If the target noun is plural, do NOT set 'limit' unless a specific number is requested.\n"
            "    - Set 'sort_order' to 'asc' and 'sort_by_field' to 'count'.\n"
            "  * For standard grouping/distribution queries without a superlative (e.g. 'each year', 'yearly', 'by state', 'by chapter', 'each month', 'each state', 'all chapters', 'all states', 'all status counts'), do NOT set 'limit' (set limit to null) and set 'sort_order' to null.\n"
            "  * If the question asks for BOTH extremes/superlatives (e.g., 'highest and lowest', 'most and least', 'top and bottom', 'maximum and minimum' count/filings), do NOT set 'limit' (set limit to null) and set 'sort_order' to 'desc' and 'sort_by_field' to 'count'. This allows the full sorted list to be returned so the user sees both ends of the spectrum.\n"
            "  * CRITICAL: Do NOT hallucinate or set a default value for 'state', 'chapter', 'date_or_year' (like defaulting to '2024' or current year), or other columns unless that specific filter value is explicitly stated in the user's question. For example, if the question is 'Which state has the most bankruptcy filings?', 'state' must be null, 'group_by_fields' must contain ['state'], 'limit' must be 1, and 'sort_order' must be 'desc'. If the question is 'Which states have the most filings?', 'state' must be null, 'group_by_fields' must contain ['state'], 'limit' must be null, and 'sort_order' must be 'desc'. If the user asks about 'Which year' or 'by year' without a specific year, 'date_or_year' must be null, 'group_by_fields' must contain ['year'], 'limit' must be 1 ONLY if it is a singular superlative (like 'which year has the highest filings'), and 'sort_order' must be 'desc' if superlative. If the user asks for general distribution (e.g., 'each year', 'by year', 'yearly breakdown') or both extremes (e.g. 'highest and lowest', 'most and least'), 'limit' must be null. If the user asks 'Which attorneys have...', 'limit' must be null, not 1 or 5.\n"
            "- CRITICAL DISAMBIGUATION between 'record_type' and 'status':\n"
            "  * 'New' is ONLY a record_type value (NEVER a status). If user mentions 'new cases', 'new filings', or 'new vs closed', set 'record_type' NOT 'status'.\n"
            "  * 'Closed' appears in BOTH record_type and status. Disambiguate by context:\n"
            "    - 'new vs closed', 'closed vs new' → set record_type to ['New', 'Closed'] and group_by_fields to ['record_type']\n"
            "    - 'active vs closed', 'dismissed vs closed' → set status (Active/Dismissed are status values)\n"
            "    - 'reopened vs closed' → set record_type to ['Reopened', 'Closed'] (Reopened is a record_type value)\n"
            "- 'without attorney' → set consumer_type to whatever type is requested, prose_indicator to 'Y', and has_no_attorney to true.\n"
            "- 'transfers' in corporate context → set status to 'Converted' and consumer_type to 'Corporate'.\n"
            "- 'held cases' → set status to 'Pending' (map 'held' to 'Pending').\n"
            "- 'details of customer XYZ' → set debtor_name to 'XYZ'.\n"
            "- VISUALIZATION WITH MULTIPLE STATUS VALUES: When the user asks for a chart/plot/bar/pie 'showing active and closed', "
            "'comparing active vs closed', 'active and closed case status', or similar:\n"
            "  * Set status to a list e.g. ['Active', 'Closed'] and set aggregation_type to 'count'.\n"
            "  * If another grouping dimension (e.g. by year, by state, by chapter) is also requested, set BOTH that dimension and status in 'group_by_fields' (e.g., ['year', 'status'] or ['state', 'status']).\n"
            "  * Otherwise (if no other grouping dimension is requested), set 'group_by_fields' to ['status'].\n"
            "- GROUP BY EXPLICIT COLUMNS: If the user question explicitly specifies columns or fields to group by (e.g., 'grouped by status and state', 'grouped by year and status', 'group by state and chapter'), you MUST extract all those fields into the 'group_by_fields' list. For example, 'grouped by status and state columns' -> group_by_fields: ['state', 'status'].\n"
            "- MULTILEVEL COMPARISONS / BREAKDOWNS: If the user question asks to compare/breakdown multiple categories/values of one column (e.g., active vs closed status, Chapter 7 vs 13, record type New vs Closed) across another grouping dimension (e.g., by year, by state, by chapter), you MUST include BOTH columns in the 'group_by_fields' list. For example:\n"
            "  * 'compare active vs closed by state' -> group_by_fields: ['state', 'status']\n"
            "  * 'comparing active vs closed cases by year' -> group_by_fields: ['year', 'status']\n"
            "  * 'compare Chapter 7 and 13 by state' -> group_by_fields: ['state', 'chapter']\n"
            "  * 'New vs Closed filings by year' -> group_by_fields: ['year', 'record_type']\n"
            "  * NEVER default to a yearwise distribution when explicit status values are mentioned unless both status and year are in group_by_fields.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example Attorney Singular Superlative:\n"
            "Question: \"Which attorney has the highest percentage of active cases?\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Active",\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["attorney_name"],\n'
            '  "limit": 1,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Attorney Plural Superlative:\n"
            "Question: \"Which attorneys have the highest number of active bankruptcy cases?\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Active",\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["attorney_name"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example State Plural Superlative:\n"
            "Question: \"Which states have the highest number of bankruptcy filings?\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["state"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example State Singular Superlative:\n"
            "Question: \"Which state has the most bankruptcy filings?\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["state"],\n'
            '  "limit": 1,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example 1:\n"
            "Question: \"P2 matchcode yearwise distribution\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": "P2",\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example 2:\n"
            "Question: \"chapterwise status count\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["chapter", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example 3:\n"
            "Question: \"active status cases in NY\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Active",\n'
            '  "chapter": null,\n'
            '  "state": "NY",\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Partnership Cases without attorney:\n"
            "Question: \"Partnership Cases without attorney\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": "Partnership",\n'
            '  "prose_indicator": "Y",\n'
            '  "has_no_attorney": true\n'
            "}\n\n"
            "Example ProSe cases of vp:\n"
            "Question: \"ProSe cases of vp\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": "VP",\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": "Y",\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example P3 and M3 total volume:\n"
            "Question: \"P3 and M3 total volume\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": ["P3", "M3"],\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Corporate transfers in NY:\n"
            "Question: \"Corporate transfers in NY\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Converted",\n'
            '  "chapter": null,\n'
            '  "state": "NY",\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": "Corporate",\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example show chapter 13 business cases:\n"
            "Question: \"show chapter 13 business cases\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": 13,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": "Business",\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Show held cases for 2024:\n"
            "Question: \"Show held cases for 2024\"\n"
            "JSON:\n"
            "{\n"
            '  "status": "Pending",\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": "2024",\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example show top 10 high risk cases:\n"
            "Question: \"show top 10 high risk cases\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "risk",\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": 10,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example details of customer XYZ:\n"
            "Question: \"details of customer XYZ\"\n"
            "JSON:\n"
            "{\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": "XYZ",\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Bar chart active and closed status:\n"
            "Question: \"Bar chart showing active and closed case status\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["status"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Active vs Closed pie chart:\n"
            "Question: \"Pie chart of active vs closed cases\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["status"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Bar chart of active and closed cases by state:\n"
            "Question: \"Bar chart of active and closed cases by state\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["state", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example compare active vs closed cases by year:\n"
            "Question: \"comapre active vs closed cases by year\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Compare active vs closed by state (normalized):\n"
            "Question: \"Show count of cases grouped by status and state columns filtered to show only status is Active or Closed\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["state", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example Compare active vs closed by year (normalized):\n"
            "Question: \"Show a bar chart of the count of cases grouped by status and year (extracted using strftime from date_filed) filtered to show only status is Active or Closed\"\n"
            "JSON:\n"
            "{\n"
            '  "status": ["Active", "Closed"],\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "count",\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year", "status"],\n'
            '  "limit": null,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Respond ONLY with a JSON object containing the extracted entities matching this exact schema:\n"
            "{\n"
            '  "status": null or string,\n'
            '  "chapter": null or integer/string,\n'
            '  "state": null or string,\n'
            '  "date_or_year": null or string,\n'
            '  "attorney_name": null or string,\n'
            '  "debtor_name": null or string,\n'
            '  "match_code": null or string or list of strings,\n'
            '  "client_name": null or string,\n'
            '  "city": null or string,\n'
            '  "trustee_name": null or string,\n'
            '  "trustee_city": null or string,\n'
            '  "sort_by_field": null or string,\n'
            '  "aggregation_type": null or string,\n'
            '  "group_by_fields": [],\n'
            '  "limit": null or integer,\n'
            '  "sort_order": null or string,\n'
            '  "record_type": null or string or list of strings,\n'
            '  "consumer_type": null or string,\n'
            '  "prose_indicator": null or string,\n'
            '  "has_no_attorney": null or boolean\n'
            "}\n"
            "Do not include markdown, code blocks, or explanation text outside the JSON."
        )
        
        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v
            
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            
        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return {}


def column_mapper(extracted_entities: dict, schema: dict, main_schema: dict = None, token_usage: dict = None) -> dict:
    """Stage 3: Column Mapper - Map extracted entities to exact database column names."""
    try:
        def _format_cols(s):
            if not s:
                return ""
            return "\n".join([f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}" for col in s.get("columns", [])])

        schema_text = f"Target Table: {schema['table_name']}\nColumns:\n" + _format_cols(schema)
        if main_schema and schema['table_name'] != main_schema['table_name']:
            schema_text += f"\n\nMain Table: {main_schema['table_name']}\nColumns:\n" + _format_cols(main_schema)

        prompt = (
            "You are a Database Column Mapping assistant for a Bankruptcy Case Management BI Dashboard.\n"
            "Your job is to map extracted user question entities/concepts to the exact, case-sensitive database column names from the schema.\n\n"
            f"DATABASE SCHEMA:\n{schema_text}\n\n"
            f"EXTRACTED ENTITIES:\n{json.dumps(extracted_entities, indent=2)}\n\n"
            "MAPPING RULES:\n"
            "- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' -> use 'status' column.\n"
            "- 'active/inactive flag' -> use 'active_status' column.\n"
            "- 'new/closed/reopened/update record stage' -> 'record_type' column.\n"
            "- 'bankruptcy chapter', 'chapter 7/11/13' -> 'chapter' column.\n"
            "- 'filing date', 'when filed' -> 'date_filed' column.\n"
            "- 'debtor name' -> use 'first_name' and 'last_name'.\n"
            "- 'state' -> 'state' column.\n"
            "- 'client', 'client name', 'client code', 'VP', 'SYF' or any client identifier -> 'client_name' column. NEVER map to 'match_code'.\n"
            "- 'attorney', 'lawyer' -> 'attorney_first_name', 'attorney_last_name', or 'attorney_dba' columns.\n"
            "- 'year', 'yearly', 'by year', 'annual' -> use 'date_filed' (prefer if available) or 'open_date'.\n"
            "- 'month', 'monthly', 'by month' -> use 'date_filed' (prefer if available) or 'open_date'.\n"
            "- 'matchcode', 'match_code' -> 'match_code' column.\n"
            "- 'record_type', 'record lifecycle', 'new/closed/reopened/update' -> 'record_type' column. When record_type contains filter values like ['New', 'Closed'], map it to 'record_type' column.\n"
            "- 'city', 'debtor city' -> 'city' column.\n"
            "- 'trustee city', 'trusty city' -> 'trustee_city' column.\n"
            "- 'trustee name', 'trusty name' -> 'trustee_name' column.\n"
            "- 'risk', 'risk score', 'match score', 'match_score' -> use 'match_score' column.\n"
            "- 'consumer_type' -> 'consumer_type' column.\n"
            "- 'prose_indicator' -> 'prose_indicator' column.\n"
            "- 'has_no_attorney' -> Keep as null (handled specially by query builder).\n"
            "- 'sort_by_field' -> Map to 'match_score' if value is 'risk' or 'score', 'date_filed' if value is 'date', or other appropriate column name. If null, map to null.\n"
            "- 'limit' and 'sort_order' -> Keep their values as-is (e.g. integer or string). Do not map them to columns.\n"
            "- 'group_by_fields' -> Only map the elements that are inside the input 'group_by_fields' list. If 'group_by_fields' is empty in the input, the output 'group_by_fields' MUST be empty [].\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example 1:\n"
            "Entities: {\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": "P2",\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["year"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n"
            "Mapping:\n"
            "{\n"
            '  "status": "status",\n'
            '  "chapter": "chapter",\n'
            '  "state": "state",\n'
            '  "date_or_year": "date_filed",\n'
            '  "attorney_name": ["attorney_first_name", "attorney_last_name", "attorney_dba"],\n'
            '  "debtor_name": ["first_name", "last_name"],\n'
            '  "match_code": "match_code",\n'
            '  "client_name": "client_name",\n'
            '  "city": "city",\n'
            '  "trustee_name": "trustee_name",\n'
            '  "trustee_city": "trustee_city",\n'
            '  "sort_by_field": null,\n'
            '  "aggregation_type": "count",\n'
            '  "group_by_fields": ["date_filed"],\n'
            '  "limit": null,\n'
            '  "sort_order": null,\n'
            '  "record_type": "record_type",\n'
            '  "consumer_type": "consumer_type",\n'
            '  "prose_indicator": "prose_indicator",\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Example 2:\n"
            "Entities: {\n"
            '  "status": null,\n'
            '  "chapter": null,\n'
            '  "state": null,\n'
            '  "date_or_year": null,\n'
            '  "attorney_name": null,\n'
            '  "debtor_name": null,\n'
            '  "match_code": null,\n'
            '  "client_name": null,\n'
            '  "city": null,\n'
            '  "trustee_name": null,\n'
            '  "trustee_city": null,\n'
            '  "sort_by_field": "risk",\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": 3,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": null,\n'
            '  "consumer_type": null,\n'
            '  "prose_indicator": null,\n'
            '  "has_no_attorney": null\n'
            "}\n"
            "Mapping:\n"
            "{\n"
            '  "status": "status",\n'
            '  "chapter": "chapter",\n'
            '  "state": "state",\n'
            '  "date_or_year": "date_filed",\n'
            '  "attorney_name": ["attorney_first_name", "attorney_last_name", "attorney_dba"],\n'
            '  "debtor_name": ["first_name", "last_name"],\n'
            '  "match_code": "match_code",\n'
            '  "client_name": "client_name",\n'
            '  "city": "city",\n'
            '  "trustee_name": "trustee_name",\n'
            '  "trustee_city": "trustee_city",\n'
            '  "sort_by_field": "match_score",\n'
            '  "aggregation_type": null,\n'
            '  "group_by_fields": [],\n'
            '  "limit": 3,\n'
            '  "sort_order": "desc",\n'
            '  "record_type": "record_type",\n'
            '  "consumer_type": "consumer_type",\n'
            '  "prose_indicator": "prose_indicator",\n'
            '  "has_no_attorney": null\n'
            "}\n\n"
            "Output a JSON object mapping each key in EXTRACTED ENTITIES to the exact database column name(s) (as a list or string, or null if no mapping is needed) or keeping its value as-is. Follow the few-shot example structure exactly.\n"
            "Respond ONLY with this JSON. Do not include markdown or explanations."
        )

        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        return json.loads(cleaned)
    except Exception as e:
        logger.warning("Column mapping failed: %s", e)
        return {}


def sql_builder(user_question: str, intent: str, extracted_entities: dict, column_mapping: dict, schema: dict, main_schema: dict = None, conversation_memory: dict = None, token_usage: dict = None, temp_table_name: str = None, temp_table_schema: dict = None) -> str:
    """Stage 4: SQL Builder - Construct the SQLite query based on intent, entities, and mapping."""
    try:
        def _format_cols(s):
            if not s:
                return ""
            return "\n".join([f"  - {col['name']} ({col.get('type', 'string')}): {col.get('description', '')}" for col in s.get("columns", [])])

        schema_desc = f"Table: {schema['table_name']}\nColumns:\n" + _format_cols(schema)
        if main_schema and schema['table_name'] != main_schema['table_name']:
            schema_desc += f"\n\nTable: {main_schema['table_name']}\nColumns:\n" + _format_cols(main_schema)

        # Build recent conversation context
        conv_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-2:]
            lines = []
            for entry in recent:
                lines.append(f"  User: {entry.get('user_question', '')}")
                if entry.get("sql_query"):
                    lines.append(f"  SQL: {entry.get('sql_query')}")
            conv_context = "RECENT CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"

        # Retrieve user preferences from LangGraph InMemoryStore
        preferences_text = ""
        if "memory_store" in st.session_state:
            store = st.session_state.memory_store
            pref_item = store.get(("default-user", "preferences"), "rules")
            if pref_item and "rules" in pref_item.value:
                preferences_text = "USER PREFERENCES (Follow these rules for SQL generation):\n" + "\n".join([f"- {r}" for r in pref_item.value["rules"]]) + "\n\n"


        # Build active result set section if temporary table is active
        temp_table_section = ""
        if temp_table_name and temp_table_schema:
            temp_cols_desc = "\n".join([
                f"  - {col['name']} ({col.get('type', 'string')})"
                for col in temp_table_schema.get("columns", [])
            ])
            temp_table_section = (
                f"ACTIVE RESULT SET / TEMPORARY TABLE:\n"
                f"Table Name: {temp_table_name}\n"
                f"Columns:\n{temp_cols_desc}\n\n"
                f"INSTRUCTIONS FOR FOLLOW-UP QUERIES:\n"
                f"1. The user's query is a follow-up question referencing the active result set stored in '{temp_table_name}'.\n"
                f"2. If you are querying the main table 'uploaded_data' (for instance, to get columns or aggregates not present in the temporary table):\n"
                f"   - If the temporary table contains 'account_number', you MUST filter the main table using: `WHERE account_number IN (SELECT account_number FROM {temp_table_name})`.\n"
                f"   - If the temporary table does not contain 'account_number' but contains columns like 'state', 'chapter', or 'status' (e.g. from an aggregation), you must filter using those matching columns: e.g. `WHERE state IN (SELECT state FROM {temp_table_name})`.\n"
                f"3. If you are querying the temporary table '{temp_table_name}' directly (which you should do if all needed columns like counts/filters/fields are in its schema), use it in the FROM clause: `FROM {temp_table_name}`.\n\n"
            )

        # Load sample data as knowledge base for grounding the SQL generation
        sample_data_str = load_sample_data_knowledge()
        sample_data_section = ""
        if sample_data_str:
            # Truncate to avoid blowing up the prompt (keep first 3000 chars)
            truncated = sample_data_str[:3000]
            sample_data_section = (
                "SAMPLE DATA (first 20 rows from the live database — use this to understand actual values, "
                "column formats, match_code distribution, date formats, and data patterns before writing SQL):\n"
                "```csv\n"
                f"{truncated}\n"
                "```\n\n"
            )
        # print("extracted_entities",json.dumps(extracted_entities, indent=2))
        # print("column_mapping",json.dumps(column_mapping, indent=2))
        # print("schema_desc",schema_desc)
        # print("INTENT",intent)
        # print("user_question",user_question)
        prompt = (
            "You are a Senior SQL Analyst specializing in SQLite query construction for bankruptcy analytics.\n"
            "Your task is to build a single valid SQLite query that answers the user's question, using the provided entity extraction and column mappings.\n\n"
            f"{sample_data_section}"
            f"{conv_context}"
            f"{preferences_text}"
            f"{temp_table_section}"
            f"USER QUESTION: \"{user_question}\"\n"
            f"INTENT: {intent}\n"
            f"EXTRACTED ENTITIES:\n{json.dumps(extracted_entities, indent=2)}\n"
            f"COLUMN MAPPINGS:\n{json.dumps(column_mapping, indent=2)}\n"
            f"DATABASE SCHEMA:\n{schema_desc}\n\n"
            "CRITICAL CONSTRUCT RULES:\n"
            "1. You MUST use only columns and tables that are listed in the schema.\n"
            "2. Table and column names are case-sensitive. Use them exactly as shown.\n"
            "3. Date columns are filtered/grouped by year using SQLite strftime, e.g. `strftime('%Y', date_column) = '2024'` or `substr(date_column, 1, 4) = '2024'`.\n"
            "4. If text filtering is done on status or other columns, use `TRIM(column_name)` to handle trailing/leading whitespace in text fields.\n"
            "5. If a distribution/breakdown of X by Y is requested (e.g., 'chapter distribution for each state'), select both columns, group by both columns, and count the total records.\n"
            "5b. If the user asks for a comparison of multiple values of a column (e.g., 'active vs closed status') by another dimension (e.g. 'by state', 'by year'), this is a multilevel comparison. You MUST select and group by BOTH columns (e.g., SELECT state, TRIM(status) AS status, COUNT(*) AS count ... GROUP BY state, status). If a year/month is involved, group by both year/month and the comparison column (e.g. GROUP BY year, status). Always select the clean values: select TRIM(status) AS status, strftime('%Y', date_filed) AS year, state, etc.\n"
            "6. Keep the query clean: do not end with a semicolon ';', and only use a SELECT statement.\n"
            "7. Use the SAMPLE DATA above to infer exact column value formats (e.g. match_code uses values like 'P1', 'P2', 'M1'; status uses 'Active', 'Closed'; date_filed is in YYYY-MM-DD format).\n"
            "8. When filtering by match_code (e.g. 'P2 matchcode'), ALWAYS add a WHERE clause: WHERE TRIM(match_code) = 'P2'. If match_code is a list (e.g. ['P3', 'M3']), use IN clause: WHERE TRIM(match_code) IN ('P3', 'M3').\n"
            "9. When filtering by client (e.g. 'VP yearwise'), ALWAYS add a WHERE clause using 'client_name' column: WHERE TRIM(client_name) = 'VP'. NEVER use match_code for client filtering.\n"
            "9b. When filtering by record_type (e.g. 'new vs closed cases'), add WHERE TRIM(record_type) IN ('New', 'Closed') and GROUP BY record_type. If record_type is a list in EXTRACTED ENTITIES, use IN with all values. If a single string, use = with that value.\n"
            "9c. When filtering by ANY field where the EXTRACTED ENTITIES provides a list (e.g., chapter: [11, 13] or status: ['Active', 'Closed']), you MUST use an IN clause (e.g., WHERE chapter IN ('11', '13') or WHERE TRIM(status) IN ('Active', 'Closed')).\n"
            "10. If a specific year (e.g., '2024') or relative timeframe (e.g., 'last 12 months', 'last year') is requested in the user query and present in `date_or_year` in EXTRACTED ENTITIES, you MUST include a WHERE filter for that time. For a specific year, use (e.g., `strftime('%Y', date_filed) = '2024'`). For a relative timeframe like 'last 12 months', use date math: `WHERE date_filed >= DATE((SELECT MAX(date_filed) FROM uploaded_data), '-12 months')`. DO NOT retrieve or aggregate over all dates when a specific time is explicitly requested. If the user asks for a distribution/breakdown within a single year (e.g., '2024 distribution') but does not specify another grouping attribute (like chapter or state), group by month (e.g. `strftime('%m', date_filed) AS month`) to show a meaningful distribution.\n"
            "11. If the user requests sorting or limiting (e.g., 'top 3 risk cases', 'highest score cases', or `limit` and `sort_by_field` are set in EXTRACTED ENTITIES), you MUST construct a SELECT query that retrieves case records (selecting relevant columns like match_score, first_name, last_name, client_name, match_code, status, date_filed, case_number), order by the mapped `sort_by_field` column (e.g. match_score for risk/score) according to the `sort_order` (e.g. `ORDER BY match_score DESC`), and apply the `limit` (e.g. `LIMIT 3`). DO NOT group or count unless specifically asked.\n"
            "11b. Pay careful attention to singular vs plural requests! If the user asks for singular (e.g. 'Which state has the most...', 'Which attorney has the highest...'), use `LIMIT 1`. If the user asks for plural without a specific number (e.g. 'Which states have the most...', 'Which attorneys have the highest...'), do NOT use any LIMIT, just use `ORDER BY count DESC`. If a general distribution or grouping over all categories is requested (e.g., 'each year', 'filings by state', 'breakdown by chapter', 'each month'), or if the user asks for BOTH extremes/superlatives (e.g. 'highest and lowest', 'most and least'), do NOT apply any simple LIMIT, but rather use UNION ALL for both extremes as specified in Rule 21. If a specific number is requested (e.g. 'top 3 states'), use that exact limit (e.g. `LIMIT 3`). Do NOT use arbitrary limits like `LIMIT 5` unless specifically requested.\n"
            "11c. CRITICAL DISTINCTION — 'top N [entity]' vs 'distribution of X for top N [parent]':\n"
            "  * Pattern A — 'top N of a GROUPED column' (e.g. 'top 3 chapters in Texas', 'top 5 attorneys by count'): The 'top N' limits the GROUP-BY result itself. Use a simple query: `SELECT chapter, COUNT(*) AS count FROM uploaded_data WHERE TRIM(state) = 'TX' GROUP BY chapter ORDER BY count DESC LIMIT 3`. Apply LIMIT N directly on the outer query.\n"
            "  * Pattern B — 'distribution of X for the top N of another column' (e.g. 'chapter distribution for top 5 states'): The 'top N' filters a PARENT category, not the breakdown. Do NOT apply LIMIT N to the outer query. Instead filter using a subquery: `SELECT state, chapter, COUNT(*) AS count FROM uploaded_data WHERE state IN (SELECT state FROM uploaded_data GROUP BY state ORDER BY COUNT(*) DESC LIMIT 5) GROUP BY state, chapter ORDER BY state, count DESC`.\n"
            "  * KEY: If the user says 'top N X in [specific filter value]' (e.g. 'top 3 chapters in Texas'), it is Pattern A — use LIMIT. If the user says 'X distribution for top N Y' or 'X by top N Y' (e.g. 'chapter distribution for top 5 states'), it is Pattern B — use subquery, no outer LIMIT.\n"
            "11d. CRITICAL: Only use Pattern B (subquery) if a distribution of one column (e.g., 'chapter') is requested for the top N of ANOTHER column (e.g., 'state'). If the query only asks for the top N of a single column (e.g., 'top 5 states with most filings', 'Which states have the highest number of filings'), it is NOT Pattern B. Use a simple GROUP BY on that column and apply the LIMIT directly to the outer query (e.g. `SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state ORDER BY count DESC LIMIT 5`).\n"
            "12. When consumer_type is specified (e.g. Partnership, Business, Corporate), filter it using `TRIM(consumer_type) = 'value'`.\n"
            "13. When prose_indicator is Y, filter using `TRIM(prose_indicator) = 'Y'`.\n"
            "14. When has_no_attorney is true, filter using: `(TRIM(prose_indicator) = 'Y' OR attorney_first_name IS NULL OR TRIM(attorney_first_name) = '')`.\n"
            "15. When debtor_name is set for a lookup query (e.g. customer name XYZ), perform a case-insensitive partial match search using: `(first_name LIKE '%XYZ%' OR last_name LIKE '%XYZ%')`.\n"
            "16. When held cases is requested, use status Pending: `TRIM(status) = 'Pending'`.\n"
            "17. When corporate transfers in NY is requested, map 'transfers' to `(TRIM(status) = 'Converted' OR (conversion_date IS NOT NULL AND conversion_date != ''))`.\n"
            "18. When high risk cases are requested, filter by `match_score >= 98`.\n"
            "19. If the RECENT CONVERSATION HISTORY is present and the current user question is a follow-up query asking for details, a breakdown, or a group-by on the previous query, you must construct a query against the main database table (uploaded_data) that inherits and applies the EXACT filters from the previous query's SQL (e.g. if the previous query was filtered to WHERE TRIM(match_code) IN ('P3', 'M3'), you must include that exact WHERE clause in your new query). CRITICAL EXCEPTIONS — do NOT inherit from previous SQL when:\n"
            "   a) The current question explicitly names its OWN filter values for a dimension (e.g., explicitly names a specific state, chapter, or status). Use ONLY the values the user named. Example: previous SQL had 'WHERE state IN (''TX'', ''AZ'')' but user now says 'chapters in Texas' → filter ONLY on TX.\n"
            "   b) A filter dimension (e.g. status='Active') appears in a prior SQL but is NOT mentioned at all in the current question. Do NOT silently carry forward status, chapter, or other filters from history unless the user explicitly references them (e.g. via 'same status', 'same filter', 'for active cases'). Example: previous SQL filtered status='Active', current question is 'top 3 chapters in Texas' with no mention of status → do NOT add status filter.\n"
            "20. When calculating 'increase', 'growth', or 'change' over time (e.g., 'largest increase over the past year'), use conditional aggregation to subtract the previous period's count from the current period's count. For example: `SUM(CASE WHEN date_filed >= DATE((SELECT MAX(date_filed) FROM uploaded_data), '-12 months') THEN 1 ELSE 0 END) - SUM(CASE WHEN date_filed >= DATE((SELECT MAX(date_filed) FROM uploaded_data), '-24 months') AND date_filed < DATE((SELECT MAX(date_filed) FROM uploaded_data), '-12 months') THEN 1 ELSE 0 END) AS increase`. Group by the requested field and sort by `increase DESC`.\n"
            "21. If the user's question asks for BOTH extremes/superlatives (e.g., 'highest and lowest', 'most and least', 'top and bottom', 'maximum and minimum' count/filings), you MUST construct a query combining the top 1 (ordered count DESC LIMIT 1) and the bottom 1 (ordered count ASC LIMIT 1) using UNION ALL. Enclose each subquery fully in parentheses. Example: `SELECT * FROM (SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state ORDER BY count DESC LIMIT 1) UNION ALL SELECT * FROM (SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state ORDER BY count ASC LIMIT 1)`.\n"
            "22. If the user asks for a percentage or count of a specific subset in the 'top' item (e.g., 'percentage of Chapter 7 cases in the top state'), do NOT use a GROUP BY on the outer query if it causes NULL rows. Instead, find the top item using a subquery and filter the main query by it, then calculate the single percentage value across that filtered set.\n"
            "23. SQLITE SYNTAX LIMITATIONS — THIS IS SQLITE, NOT MYSQL/POSTGRES:\n"
            "   - DO NOT USE: DATE_ADD, DATE_SUB, DATEDIFF, NOW(), CURDATE(), YEAR(), MONTH(), DAY(), EXTRACT()\n"
            "   - USE STRFTIME('%Y', date_column) instead of YEAR(date_column)\n"
            "   - USE STRFTIME('%m', date_column) instead of MONTH(date_column)\n"
            "   - USE DATE('now') instead of NOW() or CURDATE()\n"
            "   - USE DATE(date_column, '+30 days') or DATE(date_column, '-1 month') instead of DATE_ADD/DATE_SUB\n"
            "   - USE CAST(JULIANDAY(date_a) - JULIANDAY(date_b) AS INTEGER) instead of DATEDIFF(date_a, date_b)\n"
            "   - DO NOT USE STATISTICAL FUNCTIONS: STDDEV, VARIANCE, CORR, COVAR\n"
            "24. CRITICAL: NEVER include a WHERE clause filter for a column unless a specific filter value for that column is explicitly set to a non-null value in EXTRACTED ENTITIES. For example, if 'state' is null in EXTRACTED ENTITIES, you MUST NOT filter by state in the WHERE clause (do not use WHERE state = 'TX', WHERE TRIM(state) = 'FL', etc.). Mappings only indicate which columns exist, they do NOT justify adding filters for those columns if the entity value is null.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Please Refer to the below examples for Query Generations:\n\n"
            "Example top N chapters in state (Pattern A — LIMIT on grouped column):\n"
            "Question: \"Identify the top 3 chapters by count in Texas\"\n"
            "Entities: {\"state\": \"TX\", \"limit\": 3, \"sort_order\": \"desc\", \"sort_by_field\": \"count\", \"group_by_fields\": [\"chapter\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"state\": \"state\", \"group_by_fields\": [\"chapter\"]}\n"
            "Sample SQL: SELECT chapter, COUNT(*) AS count FROM uploaded_data WHERE TRIM(state) = 'TX' GROUP BY chapter ORDER BY count DESC LIMIT 3\n\n"
            "Example chapter distribution for top N states (Pattern B — subquery, no outer LIMIT):\n"
            "Question: \"What is the chapter distribution for the top 5 states?\"\n"
            "Entities: {\"limit\": 5, \"sort_order\": \"desc\", \"group_by_fields\": [\"state\", \"chapter\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"group_by_fields\": [\"state\", \"chapter\"]}\n"
            "Sample SQL: SELECT state, chapter, COUNT(*) AS count FROM uploaded_data WHERE state IN (SELECT state FROM uploaded_data GROUP BY state ORDER BY COUNT(*) DESC LIMIT 5) GROUP BY state, chapter ORDER BY state, count DESC\n\n"
            "Example 1:\n"
            "Question: \"P2 matchcode yearwise distribution\"\n"
            "Entities: {\"match_code\": \"P2\", \"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"match_code\": \"match_code\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data WHERE TRIM(match_code) = 'P2' GROUP BY year ORDER BY year\n\n"
            "Example VP:\n"
            "Question: \"VP yearwise distribution\"\n"
            "Entities: {\"client_name\": \"VP\", \"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' GROUP BY year ORDER BY year\n\n"
            "Example VP 2024:\n"
            "Question: \"VP 2024 distributions\"\n"
            "Entities: {\"client_name\": \"VP\", \"date_or_year\": \"2024\", \"group_by_fields\": [\"month\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"date_or_year\": \"date_filed\", \"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%m', date_filed) AS month, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' AND strftime('%Y', date_filed) = '2024' GROUP BY month ORDER BY month\n\n"
            "Example 2:\n"
            "Question: \"chapterwise status count\"\n"
            "Entities: {\"group_by_fields\": [\"chapter\", \"status\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"group_by_fields\": [\"chapter\", \"status\"]}\n"
            "Sample SQL: SELECT chapter, status, COUNT(*) AS count FROM uploaded_data GROUP BY chapter, status\n\n"
            "Example 3:\n"
            "Question: \"M1 matchcode statewise breakdown\"\n"
            "Entities: {\"match_code\": \"M1\", \"group_by_fields\": [\"state\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"match_code\": \"match_code\", \"group_by_fields\": [\"state\"]}\n"
            "Sample SQL: SELECT state, COUNT(*) AS count FROM uploaded_data WHERE TRIM(match_code) = 'M1' GROUP BY state ORDER BY count DESC\n\n"
            "Example 4:\n"
            "Question: \"show me top 3 risk cases\"\n"
            "Entities: {\"limit\": 3, \"sort_by_field\": \"risk\", \"sort_order\": \"desc\"}\n"
            "Mappings: {\"limit\": 3, \"sort_by_field\": \"match_score\", \"sort_order\": \"desc\"}\n"
            "Sample SQL: SELECT match_score, first_name, last_name, client_name, match_code, status, date_filed FROM uploaded_data ORDER BY match_score DESC LIMIT 3\n\n"
            "Example 5 (record_type — new vs closed):\n"
            "Question: \"VP closed vs new cases\"\n"
            "Entities: {\"client_name\": \"VP\", \"record_type\": [\"New\", \"Closed\"], \"group_by_fields\": [\"record_type\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"record_type\": \"record_type\", \"group_by_fields\": [\"record_type\"]}\n"
            "Sample SQL: SELECT record_type, COUNT(*) AS count FROM uploaded_data WHERE TRIM(client_name) = 'VP' AND TRIM(record_type) IN ('New', 'Closed') GROUP BY record_type ORDER BY count DESC\n\n"
            "Example 6 (status — active vs closed, NOT record_type):\n"
            "Question: \"active vs closed breakdown\"\n"
            "Entities: {\"status\": [\"Active\", \"Closed\"], \"group_by_fields\": [\"status\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"status\": \"status\", \"group_by_fields\": [\"status\"]}\n"
            "Sample SQL: SELECT status, COUNT(*) AS count FROM uploaded_data WHERE TRIM(status) IN ('Active', 'Closed') GROUP BY status ORDER BY count DESC\n\n"
            "Example Partnership Cases without attorney:\n"
            "Question: \"Partnership Cases without attorney\"\n"
            "Entities: {\"consumer_type\": \"Partnership\", \"has_no_attorney\": true}\n"
            "Mappings: {\"consumer_type\": \"consumer_type\", \"has_no_attorney\": null}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, consumer_type, prose_indicator, status, date_filed FROM uploaded_data WHERE TRIM(consumer_type) = 'Partnership' AND (TRIM(prose_indicator) = 'Y' OR attorney_first_name IS NULL OR TRIM(attorney_first_name) = '')\n\n"
            "Example Increase over time:\n"
            "Question: \"Which state had the largest increase in active cases over the past year\"\n"
            "Entities: {\"limit\": 1, \"group_by_fields\": [\"state\"], \"aggregation_type\": \"count\", \"status\": \"Active\"}\n"
            "Mappings: {\"group_by_fields\": [\"state\"], \"status\": \"status\"}\n"
            "Sample SQL: SELECT state, SUM(CASE WHEN date_filed >= DATE((SELECT MAX(date_filed) FROM uploaded_data), '-12 months') THEN 1 ELSE 0 END) - SUM(CASE WHEN date_filed >= DATE((SELECT MAX(date_filed) FROM uploaded_data), '-24 months') AND date_filed < DATE((SELECT MAX(date_filed) FROM uploaded_data), '-12 months') THEN 1 ELSE 0 END) AS increase FROM uploaded_data WHERE TRIM(status) = 'Active' GROUP BY state ORDER BY increase DESC LIMIT 1\n\n"
            "Example ProSe cases of vp:\n"
            "Question: \"ProSe cases of vp\"\n"
            "Entities: {\"client_name\": \"VP\", \"prose_indicator\": \"Y\"}\n"
            "Mappings: {\"client_name\": \"client_name\", \"prose_indicator\": \"prose_indicator\"}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, client_name, prose_indicator, status, date_filed FROM uploaded_data WHERE TRIM(client_name) = 'VP' AND TRIM(prose_indicator) = 'Y'\n\n"
            "Example P3 and M3 total volume:\n"
            "Question: \"P3 and M3 total volume\"\n"
            "Entities: {\"match_code\": [\"P3\", \"M3\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"match_code\": \"match_code\"}\n"
            "Sample SQL: SELECT COUNT(*) AS count FROM uploaded_data WHERE TRIM(match_code) IN ('P3', 'M3')\n\n"
            "Example Corporate transfers in NY:\n"
            "Question: \"Corporate transfers in NY\"\n"
            "Entities: {\"consumer_type\": \"Corporate\", \"state\": \"NY\", \"status\": \"Converted\"}\n"
            "Mappings: {\"consumer_type\": \"consumer_type\", \"state\": \"state\", \"status\": \"status\"}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, consumer_type, state, status, conversion_date FROM uploaded_data WHERE TRIM(consumer_type) = 'Corporate' AND TRIM(state) = 'NY' AND (TRIM(status) = 'Converted' OR (conversion_date IS NOT NULL AND conversion_date != ''))\n\n"
            "Example show high risk cases:\n"
            "Question: \"show high risk cases\"\n"
            "Entities: {\"sort_by_field\": \"risk\", \"sort_order\": \"desc\", \"limit\": 10}\n"
            "Mappings: {\"sort_by_field\": \"match_score\", \"sort_order\": \"desc\"}\n"
            "Sample SQL: SELECT match_score, first_name, last_name, client_name, status, date_filed FROM uploaded_data WHERE match_score >= 98 ORDER BY match_score DESC LIMIT 10\n\n"
            "Example High risk cases yearwise:\n"
            "Question: \"High risk cases yearwise\"\n"
            "Entities: {\"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\", \"sort_by_field\": \"risk\"}\n"
            "Mappings: {\"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data WHERE match_score >= 98 GROUP BY year ORDER BY year\n\n"
            "Example Top client filed chapter 7:\n"
            "Question: \"Top client filed chapter 7\"\n"
            "Entities: {\"chapter\": 7, \"group_by_fields\": [\"client_name\"], \"limit\": 1, \"sort_order\": \"desc\", \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"chapter\": \"chapter\", \"group_by_fields\": [\"client_name\"]}\n"
            "Sample SQL: SELECT client_name, COUNT(*) AS count FROM uploaded_data WHERE chapter = 7 GROUP BY client_name ORDER BY count DESC LIMIT 1\n\n"
            "Example show chapter 13 business cases:\n"
            "Question: \"show chapter 13 business cases\"\n"
            "Entities: {\"chapter\": 13, \"consumer_type\": \"Business\"}\n"
            "Mappings: {\"chapter\": \"chapter\", \"consumer_type\": \"consumer_type\"}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, consumer_type, chapter, status, date_filed FROM uploaded_data WHERE chapter = 13 AND TRIM(consumer_type) = 'Business'\n\n"
            "Example Show held cases for 2024:\n"
            "Question: \"Show held cases for 2024\"\n"
            "Entities: {\"status\": \"Pending\", \"date_or_year\": \"2024\"}\n"
            "Mappings: {\"status\": \"status\", \"date_or_year\": \"date_filed\"}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, status, date_filed FROM uploaded_data WHERE TRIM(status) = 'Pending' AND strftime('%Y', date_filed) = '2024'\n\n"
            "Example details of customer XYZ:\n"
            "Question: \"details of customer XYZ\"\n"
            "Entities: {\"debtor_name\": \"XYZ\"}\n"
            "Mappings: {\"debtor_name\": [\"first_name\", \"last_name\"]}\n"
            "Sample SQL: SELECT account_number, first_name, last_name, client_name, status, date_filed FROM uploaded_data WHERE first_name LIKE '%XYZ%' OR last_name LIKE '%XYZ%'\n\n"
            "Example Most Filings State:\n"
            "Question: \"Which state has the most bankruptcy filings?\"\n"
            "Entities: {\"sort_by_field\": \"count\", \"aggregation_type\": \"count\", \"group_by_fields\": [\"state\"], \"limit\": 1, \"sort_order\": \"desc\"}\n"
            "Mappings: {\"sort_by_field\": \"count\", \"aggregation_type\": \"count\", \"group_by_fields\": [\"state\"], \"limit\": 1, \"sort_order\": \"desc\"}\n"
            "Sample SQL: SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state ORDER BY count DESC LIMIT 1\n\n"
            "Example Top 3 Chapters Least Cases:\n"
            "Question: \"top 3 chapters with the least cases\"\n"
            "Entities: {\"sort_by_field\": \"count\", \"aggregation_type\": \"count\", \"group_by_fields\": [\"chapter\"], \"limit\": 3, \"sort_order\": \"asc\"}\n"
            "Mappings: {\"sort_by_field\": \"count\", \"aggregation_type\": \"count\", \"group_by_fields\": [\"chapter\"], \"limit\": 3, \"sort_order\": \"asc\"}\n"
            "Sample SQL: SELECT chapter, COUNT(*) AS count FROM uploaded_data GROUP BY chapter ORDER BY count ASC LIMIT 3\n\n"
            "Example Highest and Lowest years:\n"
            "Question: \"Show count of cases grouped by year (extracted from date_filed column) showing only the top 1 and bottom 1 combined\"\n"
            "Entities: {\"group_by_fields\": [\"year\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"group_by_fields\": [\"date_filed\"]}\n"
            "Sample SQL: SELECT * FROM (SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data GROUP BY year ORDER BY count DESC LIMIT 1) UNION ALL SELECT * FROM (SELECT strftime('%Y', date_filed) AS year, COUNT(*) AS count FROM uploaded_data GROUP BY year ORDER BY count ASC LIMIT 1)\n\n"
            "Example states with count greater than threshold:\n"
            "Question: \"states with count greater than 1000\"\n"
            "Entities: {\"group_by_fields\": [\"state\"], \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"group_by_fields\": [\"state\"]}\n"
            "Sample SQL: SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state HAVING COUNT(*) > 1000 ORDER BY count DESC\n\n"
            "Example Chapter Distribution for Top 5 States with active filter:\n"
            "Question: \"Show the count of cases grouped by chapter column for the top 5 states where the status is Active\"\n"
            "Entities: {\"status\": \"Active\", \"group_by_fields\": [\"state\", \"chapter\"], \"limit\": 5, \"aggregation_type\": \"count\"}\n"
            "Mappings: {\"status\": \"status\", \"group_by_fields\": [\"state\", \"chapter\"], \"limit\": 5}\n"
            "Sample SQL: SELECT state, chapter, COUNT(*) AS count FROM uploaded_data WHERE TRIM(status) = 'Active' AND state IN (SELECT state FROM uploaded_data WHERE TRIM(status) = 'Active' GROUP BY state ORDER BY COUNT(*) DESC LIMIT 5) GROUP BY state, chapter ORDER BY state, count DESC\n\n"
            "CRITICAL FINAL CHECK FOR SAFETY AND CORRECTNESS:\n"
            "1. Look at EXTRACTED ENTITIES. If a key is null (e.g. 'state' is null, 'status' is null, 'chapter' is null), you MUST NOT filter by that column in the query's WHERE clause. Mappings are general target columns, NOT filters. Only filter by a column if a specific non-null value is set for it in EXTRACTED ENTITIES.\n"
            "2. Do NOT copy state filters (like 'TX' or 'FL') or status filters (like 'Active') from the few-shot examples or sample data unless they were explicitly set to non-null values in the input EXTRACTED ENTITIES.\n"
            "3. If only a single column is requested for grouping/counting (e.g. 'state' in group_by_fields), do NOT select or group by any other columns like 'chapter' or 'status'.\n"
            "4. Only use Pattern B (subquery) if a distribution of one column (e.g., 'chapter') is requested for the top N of ANOTHER column (e.g., 'state'). For any single column superlative query (e.g. 'Which state has the highest filings', 'top 1 state', 'top 5 states'), do NOT use a subquery under any circumstances. Use a simple query with a GROUP BY and an outer LIMIT clause (e.g., SELECT state, COUNT(*) AS count FROM uploaded_data GROUP BY state ORDER BY count DESC LIMIT 1).\n\n"
            "Output ONLY a JSON object with a single key `sql` whose value is the built SQLite query string:\n"
            "{\n"
            '  "sql": "SELECT ..."\n'
            "}\n"
            "Do not include markdown or explanations."
        )

        response = call_llm_with_cache(prompt, temperature=0.0)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v

        sql_query = extract_sql_from_response(response)
        # Programmatic guard: If limit was not extracted, strip any hallucinated LIMIT clause from the end
        if extracted_entities.get("limit") is None:
            sql_query = re.sub(r"\bLIMIT\s+\d+\b\s*$", "", sql_query, flags=re.IGNORECASE).strip()
        return sql_query
    except Exception as e:
        logger.exception("SQL building failed: %s", e)
        return ""


def sql_validator(sql_query: str, user_question: str, schema: dict, main_schema: dict = None, token_usage: dict = None) -> dict:
    """Stage 5: SQL Validator - Validates the query for safety, schema alignment, and repairs errors."""
    return validate_sql_with_judge(sql_query, user_question, schema, token_usage=token_usage, main_schema=main_schema)


def execute_sqlite(sql_query: str) -> pd.DataFrame:
    """Stage 6: Execute SQLite - Execute query against database and return pandas DataFrame."""
    return execute_sql_query(sql_query)


def normalize_extracted_entities(entities: dict) -> dict:
    """Helper to programmatically normalize entities to match database schema casing conventions."""
    if not entities:
        return entities
    
    # Uppercase client_name if present
    if "client_name" in entities and entities["client_name"]:
        if isinstance(entities["client_name"], list):
            entities["client_name"] = [str(x).upper().strip() for x in entities["client_name"]]
        else:
            entities["client_name"] = str(entities["client_name"]).upper().strip()
            
    # Uppercase match_code if present
    if "match_code" in entities and entities["match_code"]:
        if isinstance(entities["match_code"], list):
            entities["match_code"] = [str(x).upper().strip() for x in entities["match_code"]]
        else:
            entities["match_code"] = str(entities["match_code"]).upper().strip()
            
    # Uppercase state if present (ensure NY, CA etc.)
    if "state" in entities and entities["state"]:
        if isinstance(entities["state"], list):
            entities["state"] = [str(x).upper().strip() if len(str(x).strip()) == 2 else str(x).title().strip() for x in entities["state"]]
        else:
            val = str(entities["state"]).strip()
            if len(val) == 2:
                entities["state"] = val.upper()
            else:
                entities["state"] = val.title()
                
    # Title case status (e.g. active -> Active, closed -> Closed)
    if "status" in entities and entities["status"]:
        if isinstance(entities["status"], list):
            entities["status"] = [str(x).title().strip() for x in entities["status"]]
        else:
            entities["status"] = str(entities["status"]).title().strip()
            
    # Title case record_type (e.g. new -> New)
    if "record_type" in entities and entities["record_type"]:
        if isinstance(entities["record_type"], list):
            entities["record_type"] = [str(x).title().strip() for x in entities["record_type"]]
        else:
            entities["record_type"] = str(entities["record_type"]).title().strip()
            
    # Title case consumer_type (e.g. business -> Business)
    if "consumer_type" in entities and entities["consumer_type"]:
        if isinstance(entities["consumer_type"], list):
            entities["consumer_type"] = [str(x).title().strip() for x in entities["consumer_type"]]
        else:
            entities["consumer_type"] = str(entities["consumer_type"]).title().strip()
            
    # Normalize prose_indicator to 'Y' or 'N'
    if "prose_indicator" in entities and entities["prose_indicator"]:
        val = str(entities["prose_indicator"]).upper().strip()
        if val in ['Y', 'YES', 'TRUE', '1']:
            entities["prose_indicator"] = 'Y'
        elif val in ['N', 'NO', 'FALSE', '0']:
            entities["prose_indicator"] = 'N'
            
    return entities


def generate_sql_from_question(user_question, schema, conversation_memory=None, token_usage=None, main_schema=None, temp_table_name=None, temp_table_schema=None):
    """Refactored pipeline to generate, validate, and execute SQL based on user question"""
    try:
        logger.info("Executing pipeline for user question: %s", user_question)
        
        # Step 1: Intent Classifier
        intent = intent_classifier(user_question, conversation_memory, token_usage)
        logger.info("Pipeline Step 1: Intent Classifier -> %s", intent)
        
        # Step 2: Entity Extractor
        entities = entity_extractor(user_question, intent, conversation_memory, token_usage)
        entities = normalize_extracted_entities(entities)
        logger.info("Pipeline Step 2: Entity Extractor -> %s", json.dumps(entities))
        
        # Step 3: Column Mapper
        mapping = column_mapper(entities, schema, main_schema, token_usage)
        logger.info("Pipeline Step 3: Column Mapper -> %s", json.dumps(mapping))
        
        # Step 4: SQL Builder
        sql_query = sql_builder(
            user_question, intent, entities, mapping, schema, main_schema, conversation_memory, token_usage,
            temp_table_name=temp_table_name,
            temp_table_schema=temp_table_schema
        )
        logger.info("Pipeline Step 4: SQL Builder -> %s", sql_query)
        
        # Step 5: SQL Validator
        validation_result = sql_validator(sql_query, user_question, schema, main_schema, token_usage)
        logger.info("Pipeline Step 5: SQL Validator -> %s", json.dumps(validation_result))
        
        # If repaired_query is provided, use it
        final_query = sql_query
        if not validation_result.get("is_valid") and validation_result.get("repaired_query"):
            final_query = validation_result["repaired_query"]
            
        # Step 6: Execute SQLite
        result_df = None
        if final_query:
            result_df = execute_sqlite(final_query)
            logger.info("Pipeline Step 6: Execute SQLite -> Success, fetched %d rows", len(result_df) if result_df is not None else 0)
        else:
            logger.warning("Pipeline Step 6: Execute SQLite skipped (Empty query)")
            
        return {
            "intent": intent,
            "entities": entities,
            "mapping": mapping,
            "sql_query": final_query,
            "validation_result": validation_result,
            "result_df": result_df
        }
    except Exception as e:
        logger.exception("Error executing generate_sql_from_question pipeline: %s", e)
        return {
            "intent": "unclear",
            "entities": {},
            "mapping": {},
            "sql_query": "",
            "validation_result": {"is_valid": False, "status": "ERROR", "explanation": str(e), "repaired_query": None},
            "result_df": None
        }


def validate_sql_with_judge(sql_query, user_question, schema, token_usage=None, main_schema=None):
    """Use an LLM as a judge to validate if the generated SQL is correct"""
    sql_upper = sql_query.strip().upper()

    ddl_dml = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'RENAME', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'REPLACE']
    for keyword in ddl_dml:
        if keyword in sql_upper:
            return {
                "is_valid": False,
                "status": "BLOCKED",
                "explanation": f"Write operation keyword '{keyword}' is forbidden. Only read SELECT operations are allowed.",
                "repaired_query": None,
            }

    if not sql_upper.startswith('SELECT'):
        return {
            "is_valid": False,
            "status": "INVALID",
            "explanation": "Query must begin with a SELECT keyword.",
            "repaired_query": None,
        }

    try:
        if main_schema and schema and schema.get('table_name') != main_schema.get('table_name'):
            table_desc = f"Main table: {main_schema['table_name']} or Temporary table: {schema['table_name']}"
            columns_desc = (
                f"Columns in Main table `{main_schema['table_name']}`:\n" +
                "\n".join([f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}" for col in main_schema['columns']]) +
                f"\n\nColumns in Temporary table `{schema['table_name']}`:\n" +
                "\n".join([f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}" for col in schema['columns']])
            )
            rule_5 = f"5. Table name must be either '{main_schema['table_name']}' or '{schema['table_name']}'"
        else:
            table_desc = f"Table: {schema['table_name']}"
            columns_desc = "\n".join([
                f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}"
                for col in schema['columns']
            ])
            rule_5 = f"5. Table name must be '{schema['table_name']}'"

        prompt = f"""You are an expert SQLite3 database validator. Your task is to judge whether the following SQL query is correct and valid.

DATABASE SCHEMA:
{table_desc}
Columns:
{columns_desc}

USER QUESTION: {user_question}

SQL QUERY TO VALIDATE:
{sql_query}

VALIDATION RULES:
1. Check if the SQL syntax is valid SQLite3
2. Check if column names are EXACT matches from the schema — same spelling AND same casing (e.g., 'status' not 'Status', 'active_status' not 'Active_Status'). If a column name does not exist in the schema, the query is INVALID.
3. Check if the query answers the user's question using the CORRECT column:
   - 'active vs closed' or 'case status' should use the 'status' column, NOT 'record_type' or 'active_status'
   - 'new vs closed', 'closed vs new', 'new cases', 'reopened', 'record stage' (New/Closed/Reopened/Update) should use 'record_type'. CRITICAL: 'New' is exclusively a record_type value, NEVER a status value. If the question mentions 'new' in the context of cases/records, it MUST use record_type, NOT status.
   - 'active/inactive flag' should use 'active_status'
   - 'partnership cases', 'business cases', 'corporate' should use the 'consumer_type' column (values: Partnership, Corporate, Business, Individual, Trust, SME).
   - 'pro se', 'prose' should use the 'prose_indicator' column (values: Y, N) with `prose_indicator = 'Y'`.
   - 'without attorney', 'no attorney' should filter on `(TRIM(prose_indicator) = 'Y' OR attorney_first_name IS NULL OR TRIM(attorney_first_name) = '')`.
   - 'transfers' in corporate context should map to conversions: `(TRIM(status) = 'Converted' OR (conversion_date IS NOT NULL AND conversion_date != ''))`.
   - 'held cases' should map to Pending: `TRIM(status) = 'Pending'`.
   - 'customer XYZ' lookup should use a partial match: `(first_name LIKE '%XYZ%' OR last_name LIKE '%XYZ%')`.
   - 'high risk cases' should filter by `match_score >= 98`.
4. Check if the query limits the results to only the specified comparison categories if the user asked for a breakdown/comparison between specific categories.
5. Check if the user's question asks for analysis, counts, or grouping 'by year' or 'yearly'. If so, the query must extract the year using strftime('%Y', date_column) or substr(date_column, 1, 4), name it `year`, include it in the SELECT list, and group by it. If the query fails to group by year when requested, it is INVALID.
6. Check if the user's question asks for a distribution, breakdown, ratio, or comparison of one column (e.g. chapter) by/for/each another column (e.g. state). If so, the query MUST select and group by BOTH columns.
6b. If the user's question asks to compare multiple categories/values of one column (e.g., 'active vs closed status') by another dimension (e.g., 'by state', 'by year', 'yearly'), the query MUST select and group by BOTH the category column (e.g. status) and the grouping dimension (e.g. state or year). If the query only selects/groups by one of them, it is INVALID.
7. Check if the date formats are correct (YYYY-MM-DD) if dates are involved.
{rule_5}
9. Do not include ';' at the end of the query
10. Reject any queries that attempt to modify data
11. Check if the user's question specifies a specific year (e.g., '2024' or '2023'). If so, the SQL query MUST contain a WHERE clause filtering by that year.
12. If the user's question requests sorting or limiting:
   - If it is a record retrieval query (listing individual cases/records), the query MUST order by `match_score` descending (or ascending if lowest is asked) and apply `LIMIT N`.
   - If it is an aggregation query (counting/grouping cases, e.g., 'Which states/chapters have the most/least cases'), the query MUST order by the aggregated count (e.g., `ORDER BY count DESC` or `ORDER BY active_cases_count DESC`) and apply the requested `LIMIT N`. Do not sort by `match_score` for aggregation/group-by queries.

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT ..."}}

Do not include any explanations, text, or additional content outside the JSON object."""

        response = call_llm_with_cache(prompt)
        if token_usage is not None:
            usage = count_token_usage(prompt, response)
            for k, v in usage.items():
                token_usage[k] = token_usage.get(k, 0) + v
        logger.info("SQL Validation Response:\n%s", response)

        is_valid = False
        explanation = "Validation response received"
        query = None

        response_text = response.strip()

        if re.search(r'\bYES\b', response_text.upper()):
            explanation = "Query is valid"
            is_valid = True
        else:
            explanation = "Query is invalid — auto-repaired"
            query = extract_sql_from_response(response_text)

        return {
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID",
            "explanation": explanation,
            "repaired_query": query if not is_valid else None,
        }
    except Exception as e:
        logger.exception("Error validating SQL with judge: %s", e)
        return {
            "is_valid": False,
            "status": "ERROR",
            "explanation": f"Validation error: {str(e)}",
            "repaired_query": None
        }


def clean_sql_for_whitespace(sql_query):
    """Add TRIM() to WHERE clauses to handle whitespace in data"""
    try:
        pattern = r"WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(['\"])([^\2]*?)\2"

        def replace_with_trim(match):
            col_name = match.group(1)
            quote = match.group(2)
            value = match.group(3)
            return f"WHERE TRIM({col_name}) = {quote}{value}{quote}"

        modified_query = re.sub(pattern, replace_with_trim, sql_query, flags=re.IGNORECASE)
        if modified_query != sql_query:
            logger.info("Added TRIM() to WHERE clause for whitespace handling")
        return modified_query
    except Exception as e:
        logger.exception("Error cleaning SQL for whitespace: %s", e)
        return sql_query


def execute_sql_query(sql_query):
    """Run SQLite read query against database"""
    try:
        sql_query = clean_sql_for_whitespace(sql_query)
        logger.info("Executing SQLite SQL: %s", sql_query)
        conn = sqlite3.connect('data.db')
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        logger.info("Executed query successfully | returned %d rows", len(df))
        return df
    except Exception as e:
        logger.exception("Database query execution failed: %s", e)
        # Do NOT call st.error() here — this function is a utility that can be
        # called outside a Streamlit render context. The caller is responsible
        # for showing error messages to the user.
        return None


def should_generate_insights(user_query, result_df):
    """Determine if a visual chart should be generated from user query."""
    if result_df is None or result_df.empty or len(result_df) < 2:
        return False
    
    # Generic visualization and chart/plot request keywords
    CHART_KEYWORDS = {
        # Direct chart/plot/visual requests
        "plot", "plots", "chart", "charts", "graph", "graphs",
        "visualize", "visualise", "visualization", "visualisation", "draw",
        # Generic request keywords that should trigger charts
        "show", "display", "view", "breakdown", "distribution", "trend", "trends",
        # Explicit chart types
        "pie chart", "bar chart", "line chart", "area chart",
        "donut chart", "scatter plot", "histogram", "heatmap",
        "horizontal bar",
        # Short type names — only when they clearly mean a chart
        "pie", "donut", "scatter",
        # Insight-specific
        "insight", "insights", "dashboard",
        # Comparison / Versus keywords
        "compare", "versus", "vs", "difference",
        # Growth / calculation / percentage keywords
        "growth", "rate", "change", "calculate", "ratio", "percent", "percentage", "increase", "decrease", "pct", "yoy"
    }
    q_lower = user_query.lower()
    return any(kw in q_lower for kw in CHART_KEYWORDS)


def detect_chart_type(user_query):
    """Detect appropriate visualization type from user query keywords"""
    q = user_query.lower()
    # Most specific matches first
    if any(k in q for k in ["donut", "doughnut"]):
        return "donut"
    if any(k in q for k in ["pie", "ratio", "proportion", "breakdown"]):
        return "pie"
    if any(k in q for k in ["scatter", "correlation"]):
        return "scatter"
    if any(k in q for k in ["heatmap", "heat map", "matrix"]):
        return "heatmap"
    if any(k in q for k in ["histogram", "frequency"]):
        return "histogram"
    if any(k in q for k in ["horizontal bar", "ranked", "ranking"]):
        return "horizontal_bar"
    if any(k in q for k in ["area chart", "area plot", "filled"]):
        return "area"
    if any(k in q for k in ["line chart", "line plot", "trend line", "over time", "trend", "growth", "year", "month", "yearly", "monthly"]):
        return "line"
    if any(k in q for k in ["bar chart", "bar plot", "bar graph", "bar", "column", "compare"]):
        return "bar"
    return "auto"


def clean_column_name(name):
    """Normalize column names to standard lowercase snake_case and resolve common typos."""
    if not name or not isinstance(name, str):
        return name
    n = name.strip().lower().replace(" ", "_")
    # Clean typos / spelling inconsistencies
    n = n.replace("attorny", "attorney")   # Attorny_* → attorney_*
    n = n.replace("lastt_name", "last_name")  # _lastt_Name → _last_name
    n = n.replace("lastt", "last")           # any remaining _lastt_ variants
    n = n.replace("addl_1", "address_line_1")
    n = n.replace("addl_2", "address_line_2")
    n = n.replace("ac_no", "account_number")
    # Map bare 'client' column header to DB column 'client_name'
    if n == "client":
        n = "client_name"
    n = n.replace("creditor_time", "creditor_meeting_time")
    n = n.replace("notification_no", "notification_number")
    n = n.replace("case_no", "case_number")
    return n


def _load_and_clean_csv(uploaded_file):
    """Read CSV, strip whitespace from columns, clean rows and map columns to clean schema"""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != '']
        df.columns = [clean_column_name(col) for col in df.columns]
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        logger.info("Loaded and cleaned CSV | shape=(%d, %d)", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.exception("Error processing uploaded CSV: %s", e)
        return None


def _make_query_cache_key(user_query, schema, conversation_memory=None):
    """Create a stable cache key for repeated user queries."""
    try:
        schema_str = json.dumps(schema, sort_keys=True)
        memory_str = json.dumps(conversation_memory or {}, sort_keys=True)
        combined = f"{user_query.strip().lower()}||{schema_str}||{memory_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    except Exception:
        return hashlib.sha256(user_query.strip().lower().encode('utf-8')).hexdigest()


def get_cached_query_result(cache_key, ttl_seconds=300):
    """Retrieve an item from the query cache if it has not expired (default: 5 minutes)."""
    try:
        if "query_cache" not in st.session_state:
            return None
        entry = st.session_state.query_cache.get(cache_key)
        if entry and isinstance(entry, dict) and "timestamp" in entry:
            elapsed = time.time() - entry["timestamp"]
            if elapsed < ttl_seconds:
                logger.info("Query cache hit | elapsed=%.1fs | key=%s", elapsed, cache_key[:8])
                return entry["value"]
            else:
                logger.info("Query cache expired | key=%s", cache_key[:8])
                del st.session_state.query_cache[cache_key]
    except Exception as e:
        logger.warning("Error reading from query cache: %s", e)
    return None


def set_cached_query_result(cache_key, value):
    """Store an item in the query cache with a timestamp."""
    try:
        if "query_cache" not in st.session_state:
            st.session_state.query_cache = {}
        st.session_state.query_cache[cache_key] = {
            "timestamp": time.time(),
            "value": value
        }
        logger.info("Stored query cache entry | key=%s", cache_key[:8])
    except Exception as e:
        logger.warning("Error writing to query cache: %s", e)


# =============================================================================
# TEMPORARY TABLE MANAGEMENT FOR FOLLOW-UP QUESTIONS
# =============================================================================

def cleanup_old_temp_tables(ttl_seconds=1800):
    """Drop temporary SQLite tables that are older than ttl_seconds.
    Also drops legacy temp tables from the database.
    """
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'temp_result_%'")
        tables = [r[0] for r in cursor.fetchall()]
        
        now = time.time()
        dropped_count = 0
        for table in tables:
            # Table pattern: temp_result_{timestamp}_{random}
            match = re.match(r"^temp_result_(\d+)", table)
            if match:
                timestamp_val = int(match.group(1))
                # If timestamp is legacy (less than 10 digits/1e9), drop it immediately to clean up bloat
                if timestamp_val < 1000000000 or (now - timestamp_val > ttl_seconds):
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                    dropped_count += 1
            else:
                # Fallback to drop other non-standard temp_result_ tables
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                dropped_count += 1
        
        if dropped_count > 0:
            conn.commit()
            logger.info("Cleaned up %d orphaned temporary tables", dropped_count)
        conn.close()
    except Exception as e:
        logger.warning("Failed to clean up old temporary tables: %s", e)


def create_temporary_table_from_dataframe(result_df, source_query):
    """Create a temporary SQLite table from a result dataframe for follow-up queries.
    
    Returns a tuple of (table_name, schema_dict) for use in follow-up queries.
    """
    try:
        # Run cleanup of old temporary tables first
        cleanup_old_temp_tables()
        
        # Create a table name with format temp_result_{timestamp}_{rand}
        ts = int(time.time())
        rand = random.randint(100, 999)
        table_name = f"temp_result_{ts}_{rand}"
        
        # Connect and create temporary table
        conn = sqlite3.connect('data.db')
        result_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        
        # Build schema for this temporary table
        temp_schema = {
            "table_name": table_name,
            "columns": [
                {
                    "name": col,
                    "type": str(result_df[col].dtype),
                    "description": f"Column from previous query result"
                }
                for col in result_df.columns
            ]
        }
        
        logger.info("Created temporary table | name=%s | rows=%d | columns=%d", 
                    table_name, len(result_df), len(result_df.columns))
        
        return table_name, temp_schema
    except Exception as e:
        logger.exception("Failed to create temporary table: %s", e)
        return None, None


def drop_temporary_table(table_name):
    """Drop a temporary table from the database."""
    if not table_name:
        return True
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        conn.close()
        logger.info("Dropped temporary table | name=%s", table_name)
        return True
    except Exception as e:
        logger.exception("Failed to drop temporary table: %s", e)
        return False


def is_chart_request(user_query):
    """Detect if the user is asking for a chart or visualization."""
    chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualise', 'visualization', 'pie', 'piechart', 'bar', 'barchart', 'line', 'trend', 'donut']
    query_lower = user_query.lower()
    return any(kw in query_lower for kw in chart_keywords)


def is_pure_chart_request(user_query):
    """Detect if the query is purely asking to chart the active result set (no new filters/sorting)."""
    query_lower = user_query.lower().strip()
    
    if not is_chart_request(user_query):
        return False
        
    filter_indicators = [
        'greater', 'less', 'more than', 'fewer than', 'above', 'below', 
        'equal', 'limit', 'top', 'bottom', 'where', 'having', 'filter',
        'sort', 'order by', 'group by', 'count greater', 'count less', 
        '>', '<', '=', '>=', '<=', 
        'compare', 'comparing', 'vs','Vs', 'versus', 'between', 
        'chapter', 'state', 'status', 'client', 'attorney', 'active', 'closed', 'match code', 'record type'
    ]
    
    has_filter = False
    for ind in filter_indicators:
        if ind in query_lower:
            if ind == 'above':
                if not any(x + ' above' in query_lower for x in ['data', 'results', 'table', 'items', 'rows', 'query', 'shown']):
                    has_filter = True
            else:
                has_filter = True
                
    if has_filter:
        return False
        
    return True


def _extract_entities_from_sql(sql_query: str) -> dict:
    """Extract key column names and filter values from a previous SQL query string.
    Used by is_followup_question to detect semantic overlap without an LLM call.

    Returns a dict with:
      - columns: set of column names referenced
      - values:  set of string literal values used in WHERE clauses
      - has_where: True if the SQL has a WHERE clause
    """
    if not sql_query:
        return {"columns": set(), "values": set(), "has_where": False}

    sql_lower = sql_query.lower()

    # Extract column names: look for identifiers before operators or in GROUP BY / ORDER BY
    col_pattern = re.compile(
        r'\b(?:where|and|or|group\s+by|order\s+by|select|trim\()\s*'
        r'([a-z_][a-z0-9_]*)',
        re.IGNORECASE,
    )
    columns = {m.group(1) for m in col_pattern.finditer(sql_lower)
               if m.group(1) not in {
                   'and', 'or', 'not', 'in', 'is', 'null', 'like', 'between',
                   'select', 'from', 'where', 'group', 'by', 'order', 'limit',
                   'having', 'as', 'on', 'join', 'left', 'right', 'inner',
               }}

    # Extract string literal values used in WHERE conditions
    value_pattern = re.compile(r"'([^']+)'")
    values = {m.group(1).lower() for m in value_pattern.finditer(sql_lower)}

    # Extract numeric literals used in WHERE conditions (chapter = 7, match_score >= 90)
    num_pattern = re.compile(r'(?:=|>=|<=|>|<|\bin\b)\s*(\d+)')
    for m in num_pattern.finditer(sql_lower):
        values.add(m.group(1))

    has_where = 'where' in sql_lower

    return {"columns": columns, "values": values, "has_where": has_where}


def is_followup_question(user_query: str, conversation_history: dict) -> bool:
    """Ultra-smart multi-signal follow-up question detector.

    Uses 6 weighted signals to decide if a query continues the current
    result-set context or starts a completely fresh query.

    Signal 1 – Explicit reference pronouns / deixis words
    Signal 2 – Conversational continuation openers
    Signal 3 – Drilldown / sub-grouping intent
    Signal 4 – Filter modification / refinement language
    Signal 5 – Semantic entity overlap with previous SQL
    Signal 6 – Comparative / ranking request on existing data

    A false-positive guard penalises the score when the query introduces
    a brand-new independent entity that was absent from the previous query.
    """
    # Guard: no active result set means nothing to follow up on
    if not st.session_state.temp_table_name:
        return False

    query_lower = user_query.lower().strip()
    words = query_lower.split()
    score = 0.0
    signals_fired = []

    # ─────────────────────────────────────────────────────────────────────────
    # Pull the last SQL query from conversation memory for semantic analysis
    # ─────────────────────────────────────────────────────────────────────────
    last_sql = ""
    last_user_q = ""
    if conversation_history and conversation_history.get("history"):
        for entry in reversed(conversation_history["history"]):
            if entry.get("sql_query"):
                last_sql = entry["sql_query"]
                last_user_q = entry.get("user_question", "")
                break

    prev_entities = _extract_entities_from_sql(last_sql)
    prev_cols = prev_entities["columns"]
    prev_vals = prev_entities["values"]

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 1 – Explicit reference pronouns / deixis (weight 0.9)
    # ─────────────────────────────────────────────────────────────────────────
    REFERENCE_WORDS = {
        'that', 'those', 'these', 'it', 'them', 'they',
        'results', 'rows', 'records', 'items', 'entries', 'data',
        'above', 'previous', 'last', 'prior', 'same', 'current',
        'selected', 'shown', 'listed', 'returned',
        'subset', 'filtered set', 'this set', 'the data',
        'these cases', 'those cases', 'those records', 'from that',
        'from there', 'among those', 'convert', 'earlier', 'aforementioned',
    }
    if any(kw in query_lower for kw in REFERENCE_WORDS):
        score += 0.9
        signals_fired.append("S1:pronouns")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 2 – Conversational continuation openers (weight 0.7)
    # ─────────────────────────────────────────────────────────────────────────
    CONTINUATION_STARTERS = [
        'and ', 'but ', 'also ', 'additionally ', 'furthermore ',
        'now ', 'now show', 'now filter', 'now give', 'what about ',
        'how about ', 'can you also', 'give me only', 'show me only',
        'narrow down', 'zoom in', 'of those', 'for them',
        'in that group', 'within those', 'among those', 'from these',
    ]
    if any(query_lower.startswith(kw) or kw in query_lower
           for kw in CONTINUATION_STARTERS):
        score += 0.7
        signals_fired.append("S2:continuation")

    # Very short queries with no full entity are almost always follow-ups
    if len(words) <= 5 and last_sql:
        score += 0.6
        signals_fired.append("S2b:short-query")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 3 – Drilldown / refinement intent (weight 0.65)
    # ─────────────────────────────────────────────────────────────────────────
    DRILLDOWN_WORDS = [
        'breakdown', 'break down', 'drill', 'detail', 'individual',
        'per ', 'split by', 'group by', 'categorize', 'categorise',
        'by status', 'by chapter', 'by state', 'by match', 'by client',
        'by attorney', 'by consumer', 'deeper', 'further', 'elaborate',
        'more detail', 'more details', 'expand', 'sub-group',
        'distribution of', 'distribution for',
    ]
    if any(kw in query_lower for kw in DRILLDOWN_WORDS):
        # Only treat as follow-up drilldown if there IS a previous query
        if last_sql:
            score += 0.65
            signals_fired.append("S3:drilldown")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 4 – Filter modification / refinement language (weight 0.7)
    # ─────────────────────────────────────────────────────────────────────────
    FILTER_MODIFIERS = [
        'only the', 'only show', 'only where', 'just the', 'just show',
        'exclude', 'remove', 'keep only', 'filter to', 'limit to',
        'where only', 'but only', 'specifically', 'restrict to',
        'with status', 'with chapter', 'with match', 'with client',
        'narrow to', 'refine', 'show only', 'display only',
    ]
    if any(kw in query_lower for kw in FILTER_MODIFIERS):
        score += 0.7
        signals_fired.append("S4:filter-modify")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 5 – Semantic entity overlap with previous SQL (weight 0.75)
    # ─────────────────────────────────────────────────────────────────────────
    if prev_cols or prev_vals:
        overlap_count = 0
        total_prev = len(prev_cols) + len(prev_vals)

        # Check if current query mentions previous column names
        for col in prev_cols:
            if col in query_lower:
                overlap_count += 1

        # Check if current query mentions previous filter values (e.g. 'VP', '7', 'P2')
        for val in prev_vals:
            if val and len(val) >= 2 and val in query_lower:
                overlap_count += 1

        if total_prev > 0:
            overlap_ratio = overlap_count / total_prev
            if overlap_ratio >= 0.4:
                score += 0.75 * min(overlap_ratio / 0.6, 1.0)
                signals_fired.append(f"S5:entity-overlap({overlap_ratio:.0%})")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 6 – Comparative / ranking on existing data (weight 0.5)
    # ─────────────────────────────────────────────────────────────────────────
    COMPARATIVE_WORDS = [
        'compare', 'vs ', 'versus', 'which is higher', 'which is more',
        'which has more', 'which is better', 'which is worse',
        'rank', 'ranking', 'highest', 'lowest', 'best', 'worst',
        'most', 'least', 'difference between', 'compared to',
    ]
    if any(kw in query_lower for kw in COMPARATIVE_WORDS) and last_sql:
        score += 0.5
        signals_fired.append("S6:comparative")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart-only re-render shortcut (pure visualization of active result set)
    # ─────────────────────────────────────────────────────────────────────────
    if is_chart_request(user_query):
        CHART_PIVOT_HINTS = [
            'year', 'month', 'quarter', 'state', 'district', 'chapter',
            'attorney', 'debtor', 'filing', 'judge', 'trustee',
            'by state', 'by year', 'by chapter', 'by district',
            'by attorney', 'over time', 'trend',
        ]
        if any(hint in query_lower for hint in CHART_PIVOT_HINTS):
            # Chart requests a new grouping dimension → treat as fresh query
            logger.debug("Follow-up suppressed: chart request introduces new pivot dimension")
            return False
        # Pure re-chart with no new dimension → direct follow-up
        logger.debug("Follow-up detected: pure chart re-render of active result set")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # FALSE-POSITIVE GUARD – penalise if a wholly new entity is introduced
    # or if the query is a request for a fresh complete distribution ("all", "each", "every")
    # ─────────────────────────────────────────────────────────────────────────
    # Detect brand-new named entity references in the current query
    penalty = 0.0

    # Fresh distribution request guard
    FRESH_DISTRIBUTION_PATTERNS = [
        r'\ball\s+chapters\b', r'\ball\s+states\b', r'\ball\s+status\b', r'\ball\s+match\b', r'\ball\s+client\b', r'\ball\s+attorney\b',
        r'\beach\s+chapter\b', r'\beach\s+state\b', r'\beach\s+status\b', r'\beach\s+match\b', r'\beach\s+client\b', r'\beach\s+attorney\b',
        r'\bevery\s+chapter\b', r'\bevery\s+state\b', r'\bevery\s+status\b', r'\bevery\s+match\b', r'\bevery\s+client\b', r'\bevery\s+attorney\b',
        r'\bshow\s+all\b', r'\blist\s+all\b', r'\bcount\s+all\b',
    ]
    for pattern in FRESH_DISTRIBUTION_PATTERNS:
        if re.search(pattern, query_lower):
            penalty += 0.8
            signals_fired.append("G:fresh-distribution-request")
            break
    US_STATES = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
        'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
        'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
        'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
        'new hampshire', 'new jersey', 'new mexico', 'new york',
        'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
        'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
        'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
        'west virginia', 'wisconsin', 'wyoming',
        # Common abbreviations
        'ny', 'ca', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi',
        'nj', 'va', 'wa', 'az', 'ma', 'tn', 'in', 'mo', 'md', 'wi',
        'mn', 'co', 'al', 'sc', 'la', 'ky', 'or', 'ok', 'ct', 'ut',
    }

    # New year that wasn't in previous query
    year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', query_lower)
    prev_years = re.findall(r'\b(20\d{2}|19\d{2})\b', last_sql.lower())
    new_years = set(year_matches) - set(prev_years)
    if new_years:
        penalty += 0.35
        signals_fired.append(f"G:new-year({','.join(new_years)})")

    # New state reference
    query_words_set = set(words)
    prev_state_in_sql = any(s in last_sql.lower() for s in US_STATES)
    new_state_in_query = any(s in query_lower for s in US_STATES)
    if new_state_in_query and not prev_state_in_sql:
        penalty += 0.35
        signals_fired.append("G:new-state")

    # New chapter number that wasn't in previous SQL
    chapter_in_q = re.findall(r'\bchapter\s+(\d+)\b', query_lower)
    chapter_in_prev = re.findall(r'\bchapter\s*=?\s*(\d+)\b', last_sql.lower())
    new_chapters = set(chapter_in_q) - set(chapter_in_prev)
    if new_chapters and not chapter_in_prev:
        penalty += 0.3
        signals_fired.append(f"G:new-chapter({','.join(new_chapters)})")

    # Apply penalty
    score -= penalty

    # ─────────────────────────────────────────────────────────────────────────
    # DECISION
    # ─────────────────────────────────────────────────────────────────────────
    is_followup = score >= 0.65

    logger.debug(
        "Follow-up detection | query=%r | score=%.2f | signals=%s | result=%s",
        user_query[:80], score, signals_fired, is_followup,
    )

    return is_followup



# =============================================================================
# CONVERSATION MEMORY FUNCTIONS (LANGGRAPH IN-MEMORY STORE)
# =============================================================================

import uuid
from langgraph.store.memory import InMemoryStore

def _mock_embed(texts: list) -> list:
    # Return one embedding vector per input text (required shape for InMemoryStore)
    return [[1.0, 2.0] for _ in texts]

def _initialize_conversation_memory():
    """Initialize conversation memory structure and LangGraph InMemoryStore"""
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = InMemoryStore(index={"embed": _mock_embed, "dims": 2})
        # Set up default user preferences/rules
        st.session_state.memory_store.put(
            ("default-user", "preferences"),
            "rules",
            {
                "rules": [
                    "User prefers concise, data-driven answers",
                    "Do not repeat the exact same type of follow-up questions",
                    "Make follow-up questions extremely simple, fresh, and contextual",
                    "When returning 'top item' aggregates, do not return queries that output mostly NULL rows."
                ]
            }
        )

    return {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
        "session_id": str(uuid.uuid4())
    }


def _append_conversation_memory(user_query, sql_query, records, validation_result=None, answer=None):
    """Append structured conversation memory to session state and LangGraph Store"""
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory
        store = st.session_state.memory_store
        session_id = memory.get("session_id", "default-session")
        namespace = ("default-user", "conversation", session_id)

        if not answer:
            memory_entry = {
                "user_question": user_query,
                "sql_query": sql_query,
                "records": records,
                "record_count": len(records) if records else 0,
            }
        else:
            memory_entry = {
                "user_question": user_query,
                "assistant_answer": answer,
            }

        if validation_result:
            memory_entry["validation_result"] = validation_result.get("status", "UNKNOWN")

        # Save to LangGraph InMemoryStore
        interaction_id = str(uuid.uuid4())
        store.put(namespace, interaction_id, memory_entry)

        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query

        if not answer:
            memory["last_assistant_response"] = f"SQL: {sql_query} | Records: {len(records) if records else 0}"
        else:
            memory["last_assistant_response"] = answer

        # Keep only last 10 conversations in standard UI history
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]

    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)


# =============================================================================
# USER QUERY HANDLER & CHAT INTERACTION
# =============================================================================

def _handle_user_query(user_query, schema, active_schema):
    """Handle user queries by generating, validating, executing SQL and generating insights"""
    try:
        logger.info("Processing user query: %s", user_query)

        # ── Smart Query Understanding (Haiku pre-processor) ────────────────────
        schema_for_understanding = active_schema if active_schema else schema
        understanding = smart_query_understanding(
            user_query,
            schema_for_understanding,
            st.session_state.get("conversation_memory"),
        )
        logger.info(
            "Query understanding | intent=%s | answerable=%s | normalized=%s",
            understanding["intent"],
            understanding["is_answerable"],
            understanding["normalized_query"][:120],
        )

        # If the query is deemed unanswerable from schema, surface clarification
        if not understanding["is_answerable"] and understanding["clarification_needed"]:
            clarification_msg = (
                f"I wasn't able to find a direct match for that in the dataset. "
                f"{understanding['clarification_needed']}\n\n"
                "💡 *Try rephrasing using column names like* `status`, `chapter`, `state`, `client_name`, or `match_code`."
            )
            # Show in chat thread (not a floating warning box)
            st.markdown(clarification_msg)
            _append_conversation_memory(user_query, "", None, answer=clarification_msg)
            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
            return

        # Use the normalized query for all downstream processing
        effective_query = understanding["normalized_query"] if understanding["normalized_query"] else user_query
        # ── End Smart Query Understanding ──────────────────────────────────────

        # Check if it is a suggested followup query or matches the followup heuristics
        is_suggested_followup = st.session_state.get("is_suggested_followup", False)
        # Guard: safely read conversation_memory before calling is_followup_question
        _conv_mem = st.session_state.get("conversation_memory") or {}
        is_followup = (
            is_suggested_followup
            or is_followup_question(user_query, _conv_mem)
            or (understanding.get("intent") == "follow_up")
        )
        st.session_state.is_suggested_followup = False
        logger.info("Check if its asking a Follow-up question | is_followup=%s | is_suggested_followup=%s", is_followup, is_suggested_followup)

        if "followup_level" not in st.session_state:
            st.session_state.followup_level = 0

        if is_followup:
            st.session_state.followup_level += 1
            logger.info("Follow-up level: %d / 2", st.session_state.followup_level)
            if st.session_state.followup_level > 2:
                logger.info("Follow-up level exceeds 2. Resetting memory and clearing active dataset.")
                st.session_state.conversation_memory = _initialize_conversation_memory()
                if st.session_state.temp_table_name:
                    drop_temporary_table(st.session_state.temp_table_name)
                    st.session_state.temp_table_name = None
                    st.session_state.temp_table_schema = None
                    st.session_state.temp_table_source_query = None
                    st.session_state.temp_table_dataframe = None
                st.session_state.followup_level = 0
                is_followup = False
                
        else:
            st.session_state.followup_level = 0
            logger.info("Intent is a fresh query. Smartly resetting conversation memory and active dataset.")
            st.session_state.conversation_memory = _initialize_conversation_memory()
            if st.session_state.temp_table_name:
                drop_temporary_table(st.session_state.temp_table_name)
                st.session_state.temp_table_name = None
                st.session_state.temp_table_schema = None
                st.session_state.temp_table_source_query = None
                st.session_state.temp_table_dataframe = None
        
        # Determine if we should query the temporary table or fallback to the main database
        use_temp_table = False
        
        # Check if user question explicitly asks for a fresh complete distribution ("all", "each", "every")
        # in which case we should bypass the temporary table and query the main database
        is_fresh_distribution_request = False
        query_lower = user_query.lower().strip()
        FRESH_DISTRIBUTION_WORDS = ['all', 'each', 'every', 'entire', 'whole', 'fresh', 'reset']
        if any(w in query_lower for w in FRESH_DISTRIBUTION_WORDS):
            is_fresh_distribution_request = True
            
        if is_followup and st.session_state.temp_table_schema and not is_fresh_distribution_request:
            temp_cols = {col['name'].lower() for col in st.session_state.temp_table_schema.get("columns", [])}
            needed_cols = {col.lower() for col in understanding.get("relevant_columns", [])}
            
            # If the temporary table contains all columns needed, query it directly
            if needed_cols and needed_cols.issubset(temp_cols):
                use_temp_table = True
            
        if use_temp_table:
            working_schema = st.session_state.temp_table_schema
            # st.info(" **Querying previous result set**")
            logger.info("Using temporary table for follow-up query | table=%s", working_schema.get('table_name'))
        else:
            working_schema = active_schema if active_schema else schema
            if is_followup:
                logger.info("Querying main database for follow-up query using conversation history.")
            if any(kw in user_query.lower() for kw in ['clear', 'reset', 'new query', 'fresh', 'different']):
                st.session_state.conversation_memory = _initialize_conversation_memory()
                st.session_state.followup_level = 0
                if st.session_state.temp_table_name:
                    drop_temporary_table(st.session_state.temp_table_name)
                    st.session_state.temp_table_name = None
                    st.session_state.temp_table_schema = None
                    st.session_state.temp_table_source_query = None
                    st.session_state.temp_table_dataframe = None
                    st.info("Cleared previous result set. Starting fresh...")
        
        generation_token_usage = {}
        validation_token_usage = {}
        token_usage = {
            "generation": generation_token_usage,
            "validation": validation_token_usage,
        }

        cache_key = _make_query_cache_key(user_query, working_schema, st.session_state.conversation_memory)
        cached_response = get_cached_query_result(cache_key)

        if cached_response is not None:
            st.success(" Cached result found for repeated query")
            render_centered_table(cached_response["dataframe"], user_query=user_query)
            if cached_response.get("has_insights"):
                with st.expander(" View Insights and Charts", expanded=True):
                    generate_insights(
                        cached_response["dataframe"],
                        chart_type=cached_response.get("chart_type", "auto"),
                        user_query=user_query,
                    )
            _append_conversation_memory(
                user_query,
                cached_response["sql_query"],
                cached_response["records"],
                cached_response.get("validation_result"),
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": cached_response["message"],
                "sql_query": cached_response["sql_query"],
                "validation_result": cached_response.get("validation_result"),
                "dataframe": cached_response["dataframe"],
                "user_query": user_query,
                "has_insights": cached_response.get("has_insights", False),
                "chart_type": cached_response.get("chart_type", "auto"),
                "token_usage": cached_response.get("token_usage", {}),
            })
            return

        is_visualization_fallback = False
        main_db_schema = active_schema if active_schema else schema
        if is_followup and st.session_state.temp_table_dataframe is not None and is_pure_chart_request(user_query):
            logger.info("Pure chart request detected. Bypassing SQL generation and reusing active dataset.")
            final_query = st.session_state.temp_table_source_query
            result_df = st.session_state.temp_table_dataframe
            validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
            is_visualization_fallback = True
            sql_query = None

        if not is_visualization_fallback:
            with st.spinner(""):
                pipeline_result = generate_sql_from_question(
                    effective_query,
                    working_schema,
                    st.session_state.conversation_memory,
                    token_usage=generation_token_usage,
                    main_schema=main_db_schema,
                )
            
            sql_query = pipeline_result["sql_query"]
            validation_result = pipeline_result["validation_result"]
            result_df = pipeline_result["result_df"]

            if not sql_query and is_followup and st.session_state.temp_table_dataframe is not None and is_chart_request(user_query):
                logger.info("Empty SQL generated for chart follow-up query. Re-using active dataset.")
                final_query = st.session_state.temp_table_source_query
                result_df = st.session_state.temp_table_dataframe
                validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
                is_visualization_fallback = True

        if not is_visualization_fallback:
            if not sql_query:
                error_msg = (
                    "I couldn't translate your question into a database query. "
                    "Please try rephrasing — for example, try being more specific about the column, "
                    "filter value, or time period you're interested in."
                )
                st.markdown(error_msg)
                _append_conversation_memory(user_query, "", None, answer=error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return

            final_query = sql_query

            if not validation_result['is_valid']:
                if validation_result.get('repaired_query'):
                    # Silently use repaired query — no need to alarm the user
                    final_query = validation_result['repaired_query']
                    logger.info("Using auto-repaired SQL query: %s", final_query)
                else:
                    error_msg = (
                        "I ran into a problem generating a valid query for your request. "
                        f"Detail: *{validation_result['explanation']}*\n\n"
                        "💡 Try rephrasing your question or specifying a different filter."
                    )
                    st.markdown(error_msg)
                    _append_conversation_memory(user_query, sql_query, None, validation_result, answer=error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sql_query": sql_query,
                        "validation_result": validation_result
                    })
                    return

            if result_df is None:
                error_msg = (
                    "The query ran but returned no data — this may be a database connectivity issue. "
                    "Please try again, or rephrase your question with different filters."
                )
                st.markdown(error_msg)
                _append_conversation_memory(user_query, final_query, None, validation_result, answer=error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sql_query": final_query,
                    "validation_result": validation_result
                })
                return
        
        if len(result_df) == 0:
            # Build a smart, context-aware no-results message
            suggestions = []
            q_lower = user_query.lower()
            if any(yr in q_lower for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]):
                suggestions.append("removing the year filter to see all available periods")
            if any(kw in q_lower for kw in ["active", "closed", "pending", "discharged"]):
                suggestions.append("trying a different status value (e.g. `Active`, `Closed`, `Pending`, `Discharged`)")
            if any(kw in q_lower for kw in ["chapter 7", "chapter 11", "chapter 13"]):
                suggestions.append("checking that the chapter number exists in the dataset")
            if any(kw in q_lower for kw in [" ny", " ca", " tx", " fl"]):
                suggestions.append("verifying the state abbreviation is uppercase (e.g. `NY`, `CA`)")
            hint = (
                "\n\n💡 **Suggestions:** " + " · ".join(f"Try {s}" for s in suggestions)
                if suggestions else ""
            )
            msg = f"No records matched your criteria, please try another query.{hint}"
            st.markdown(msg)
            _append_conversation_memory(user_query, final_query, [], validation_result, answer=msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "user_query": user_query,
                "has_insights": False,
                "chart_type": "auto",
                "token_usage": token_usage,
            })
            set_cached_query_result(cache_key, {
                "message": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "records": [],
                "user_query": user_query,
                "has_insights": False,
                "chart_type": "auto",
                "token_usage": token_usage,
            })
            return
        
        # ── Generate a concise natural-language insight headline ─────────────
        try:
            if len(result_df) <= 50:
                sample_data_for_llm = result_df.to_dict('records')
                sample_desc = "Complete result data"
            else:
                sample_data_for_llm = result_df.head(10).to_dict('records')
                sample_desc = "Sample data (first 10 rows only)"

            _headline_prompt = (
                f"You are a data analyst assistant. The user asked: \"{user_query}\".\n"
                f"The query returned a total of {len(result_df)} rows with columns: {list(result_df.columns)}.\n"
                f"{sample_desc}: {sample_data_for_llm}\n\n"
                "Write a single, concise sentence (max 20 words) summarising the key finding from these results. "
                "Be specific — include the top value, count, or percentage if visible. "
                "CRITICAL: Ground your summary numbers strictly on the total row count and complete dataset, not just the sample rows. "
                "For example, if the complete data has 4 rows (representing 4 unique clients), report 4 unique clients. "
                "Do NOT start with 'The query' or 'Here are'. Respond ONLY with that one sentence."
            )
            _headline = call_llm_haiku(_headline_prompt)
            _headline = _headline.strip().strip('"').strip("'")
            if _headline:
                success_msg = _headline
            else:
                success_msg = f"Found **{len(result_df):,}** record(s) matching your query."
        except Exception:
            success_msg = f"Found **{len(result_df):,}** record(s) matching your query."
        # ─────────────────────────────────────────────────────────────────────

        if is_visualization_fallback:
            success_msg = f"Here's the chart for the active dataset ({len(result_df):,} records)."

        st.markdown(success_msg)
        render_centered_table(result_df, user_query=user_query)
        if result_df is not None and not result_df.empty:
            old_temp_table = st.session_state.get("temp_table_name")
            temp_table_name, temp_schema = create_temporary_table_from_dataframe(result_df, final_query)
            if temp_table_name and temp_schema:
                if old_temp_table and old_temp_table != temp_table_name:
                    drop_temporary_table(old_temp_table)
                st.session_state.temp_table_name = temp_table_name
                st.session_state.temp_table_schema = temp_schema
                st.session_state.temp_table_source_query = final_query
                st.session_state.temp_table_dataframe = result_df
                st.caption(f"💾 {len(result_df):,} records stored — ask follow-up questions about this result set.")
        
        # Use original user_query for intent/keyword detection — effective_query may be schema-rewritten
        should_insights = should_generate_insights(user_query, result_df) or is_visualization_fallback
        # Only trust Haiku's 'visualization' intent if the user also used an
        # explicit chart keyword — prevents false positives on 'show me X' queries
        if not should_insights and understanding.get("intent") == "visualization":
            EXPLICIT_CHART_WORDS = {
                "plot", "chart", "graph", "visualize", "visualise",
                "pie", "bar", "donut", "scatter", "histogram",
                "heatmap", "line chart", "area chart",
            }
            if any(cw in user_query.lower() for cw in EXPLICIT_CHART_WORDS):
                should_insights = True
        chart_type = detect_chart_type(user_query)
        
        if should_insights and (len(result_df) >= 2 or is_visualization_fallback):
            with st.expander(" View Insights and Charts", expanded=True):
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        _append_conversation_memory(effective_query, final_query, result_df.to_dict('records'), validation_result)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
        })
        set_cached_query_result(cache_key, {
            "message": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "records": result_df.to_dict('records'),
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
        })
        
        logger.info("Query processed successfully | rows=%d | chart_type=%s | temp_table=%s", 
                    len(result_df), chart_type, st.session_state.temp_table_name or "none")
        
    except Exception as e:
        error_msg = (
            "Something went wrong while processing your request. "
            "Please try again or rephrase your question."
        )
        logger.exception("Error in user query handling: %s", e)
        st.markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})



def render_query_box_tab(schema, active_schema):
    """Main rendering entrypoint for the Query Box tab in app.py"""
    
    # Initialize suggested question states
    if "suggested_question_selected" not in st.session_state:
        st.session_state.suggested_question_selected = None

    chat_box = st.container(height=700)

    with chat_box:
        for idx, msg in enumerate(st.session_state.messages):
            avatar_val = "blank.png" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar_val):
                token_info_state_key = f"token_info_state_{idx}"
                token_info_button_key = f"token_info_btn_{idx}"
                if token_info_state_key not in st.session_state:
                    st.session_state[token_info_state_key] = False
                show_token_usage = st.session_state[token_info_state_key]

                content = msg.get("content")
                if isinstance(content, dict):
                    if content.get("type") == "table":
                        st.markdown(content.get("message", "Result:"))
                        df_res = pd.DataFrame(content.get("data", []))
                        render_centered_table(df_res, user_query=msg.get("user_query"))
                    elif content.get("type") == "text":
                        st.markdown(content.get("message", ""), unsafe_allow_html=True)
                    else:
                        st.write(content)
                else:
                    st.markdown(content, unsafe_allow_html=True)

                if msg["role"] == "assistant":
                    if "dataframe" in msg and msg["dataframe"] is not None:
                        render_centered_table(msg["dataframe"], user_query=msg.get("user_query"))

                    if msg.get("has_insights") and msg.get("user_query"):
                        with st.expander("View Insights and Charts", expanded=True):
                            generate_insights(
                                msg["dataframe"],
                                chart_type=msg.get("chart_type", "auto"),
                                user_query=msg["user_query"]
                            )

                    if msg["role"] == "assistant" and "token_usage" in msg and msg["token_usage"]:
                        cols = st.columns([0.92, 0.08])
                        with cols[1]:
                            if st.button("Info", key=token_info_button_key, help="Click to show token usage details"):
                                st.session_state[token_info_state_key] = not st.session_state[token_info_state_key]
                                show_token_usage = st.session_state[token_info_state_key]

                    if show_token_usage and "token_usage" in msg and msg["token_usage"]:
                        with st.expander("Token Usage", expanded=True):
                            for step_name, usage in msg["token_usage"].items():
                                if isinstance(usage, dict) and usage:
                                    st.markdown(
                                        f"**{step_name.title()}**: Input `{usage['input_tokens']}` | Output `{usage['output_tokens']}` | Total `{usage['total_tokens']}`"
                                    )

        # Process new query directly inside chat_box so the spinner displays there (below the user question)
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            query_text = last_msg["content"]

            with st.chat_message("assistant", avatar="blank.png"):
                with st.spinner("Processing request..."):
                    _handle_user_query(query_text, schema, active_schema)
                st.rerun()

        # -------------------------------------------------------------------------
        # DYNAMIC AUTO PROMPT SUGGESTIONS (TEXTUAL & VISUAL) inside the chat_box
        # -------------------------------------------------------------------------
        schema_to_use = active_schema if active_schema else schema

        # Calculate state key to avoid regenerating suggestions on every rerun
        history_len = len(st.session_state.conversation_memory.get("history", [])) if "conversation_memory" in st.session_state else 0
        active_temp_table = st.session_state.get("temp_table_name")
        last_msg_content = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        current_state_key = f"sug_{history_len}_{active_temp_table}_{st.session_state.get('last_uploaded_file_name')}_{hashlib.sha256(str(last_msg_content).encode()).hexdigest()}"

        if "suggestions_state_key" not in st.session_state or st.session_state.suggestions_state_key != current_state_key or "current_suggestions" not in st.session_state:
            with st.spinner(" Updating suggested prompts..."):
                suggestions = _generate_dynamic_prompt_suggestions(
                    schema_to_use,
                    st.session_state.get("conversation_memory"),
                    last_result_df=st.session_state.get("temp_table_dataframe"),
                )
                st.session_state.current_suggestions = suggestions
                st.session_state.suggestions_state_key = current_state_key
        else:
            suggestions = st.session_state.current_suggestions

        # CSS to style suggestion buttons as clean outline buttons (removing blue boxes)
        st.markdown(
            """
            <style>
            div:has(.suggest-anchor-textual) button,
            div:has(.suggest-anchor-visual) button,
            div:has(.suggest-anchor-textual) .stButton > button,
            div:has(.suggest-anchor-visual) .stButton > button,
            div[data-testid="column"]:has(.suggest-anchor-textual) button,
            div[data-testid="column"]:has(.suggest-anchor-visual) button {
                background-color: #ffffff !important;
                color: #475569 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
                font-weight: 500 !important;
                padding: 0.5rem 1rem !important;
                transition: all 0.2s ease !important;
                text-align: left !important;
                white-space: normal !important;
                word-wrap: break-word !important;
                height: auto !important;
                min-height: 44px !important;
                display: inline-block !important;
            }

            div:has(.suggest-anchor-textual) button:hover,
            div:has(.suggest-anchor-visual) button:hover,
            div:has(.suggest-anchor-textual) .stButton > button:hover,
            div:has(.suggest-anchor-visual) .stButton > button:hover,
            div[data-testid="column"]:has(.suggest-anchor-textual) button:hover,
            div[data-testid="column"]:has(.suggest-anchor-visual) button:hover {
                background-color: #f8fafc !important;
                border-color: #3b82f6 !important;
                color: #2563eb !important;
                box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.05) !important;
                transform: translateY(-1px);
            }

            /* Push suggestions block to the bottom of the flex container */
            div[data-testid="stHorizontalBlock"]:has(.suggest-anchor-textual) {
                margin-top: auto !important;
                padding-top: 1.5rem !important;
            }

            /* Stretch inner vertical block to fill container height to allow flex margin push */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.suggest-anchor-textual) > div[data-testid="stVerticalBlock"] {
                min-height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        has_textual = len(suggestions.get("textual", [])) > 0
        has_visual = len(suggestions.get("visual", [])) > 0

        if has_textual or has_visual:
            col_t, col_v = st.columns([1, 1])

            with col_t:
                st.markdown('<div class="suggest-anchor-textual"></div>', unsafe_allow_html=True)
                st.caption(" Suggestion Chips")
                for idx, txt_q in enumerate(suggestions.get("textual", [])):
                    if st.button(txt_q, key=f"suggest_text_{idx}", use_container_width=True):
                        st.session_state.suggested_question_selected = txt_q
                        st.session_state.is_suggested_followup = (history_len > 0)
                        st.rerun()

            with col_v:
                st.markdown('<div class="suggest-anchor-visual"></div>', unsafe_allow_html=True)
                st.caption(" Chart suggestions")
                for idx, vis_q in enumerate(suggestions.get("visual", [])):
                    if st.button(vis_q, key=f"suggest_visual_{idx}", use_container_width=True):
                        st.session_state.suggested_question_selected = vis_q
                        st.session_state.is_suggested_followup = (history_len > 0)
                        st.rerun()

            st.write("")  # spacer

    if st.session_state.temp_table_name and st.session_state.temp_table_dataframe is not None:
        temp_df = st.session_state.temp_table_dataframe
        cols = st.columns([0.96, 0.04])
        with cols[0]:
            st.markdown(
                f"**Active Result Set**: {len(temp_df)} records from previous query | "
                f"{len(temp_df.columns)} columns"
            )
        with cols[1]:
            st.markdown(
                """
                <style>
                div[data-testid="column"]:has(.clear-btn-anchor) button {
                    opacity: 0.35 !important;
                    transition: opacity 0.2s ease-in-out !important;
                    background-color: transparent !important;
                    border: 1px solid #cbd5e1 !important;
                    color: #64748b !important;
                    border-radius: 6px !important;
                    padding: 0 !important;
                    font-size: 1.15rem !important;
                    width: 28px !important;
                    height: 28px !important;
                    min-width: 28px !important;
                    max-width: 28px !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    margin-top: -6px !important;
                }
                div[data-testid="column"]:has(.clear-btn-anchor) button:hover {
                    opacity: 0.9 !important;
                    border-color: #94a3b8 !important;
                    color: #1e293b !important;
                    background-color: #f1f5f9 !important;
                }
                </style>
                <div class="clear-btn-anchor"></div>
                """,
                unsafe_allow_html=True
            )
            if st.button("↺", key="clear_temp_table", help="Clear & Reset current dataset"):
                drop_temporary_table(st.session_state.temp_table_name)
                st.session_state.temp_table_name = None
                st.session_state.temp_table_schema = None
                st.session_state.temp_table_source_query = None
                st.session_state.temp_table_dataframe = None
                st.session_state.conversation_memory = _initialize_conversation_memory()
                st.session_state.followup_level = 0
                st.rerun()

    # Replace chat input up-arrow with 'Ask' text via CSS
    st.markdown(
        """
        <style>
        button[data-testid="stChatInputSubmitButton"] svg {
            display: none;
        }
        button[data-testid="stChatInputSubmitButton"]::after {
            content: "Ask";
            font-weight: 600;
        }
        button[data-testid="stChatInputSubmitButton"] {
            width: auto !important;
            padding: 0 12px !important;
            border-radius: 4px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    user_input = st.chat_input("Query database (e.g. 'Show total records by state')...")

    # If suggested prompt was clicked, override user_input
    if st.session_state.get("suggested_question_selected"):
        user_input = st.session_state.suggested_question_selected
        st.session_state.suggested_question_selected = None

    if user_input:
        if not st.session_state.data_in_db:
            st.error("No active dataset found in database. Please upload a CSV first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

