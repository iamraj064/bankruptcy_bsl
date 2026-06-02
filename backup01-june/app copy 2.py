import sqlite3
import logging
import sys
import json
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from config import call_llm, call_llm_haiku
from dotenv import load_dotenv
from insights_generator import generate_insights

# =============================================================================
# CONFIG & INITIALIZATION
# =============================================================================
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
    page_title="Bankruptcy GenBI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PREMIUM STYLING (Slate & Blue Executive Theme + Sidebar Nav)
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

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
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_schema():
    """Load default schema from schema.json"""
    try:
        with open("schema.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None


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
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        conn.close()

        if not columns_info:
            return None

        actual_columns = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2] or "text"
            actual_columns.append({
                "name": col_name,
                "type": col_type,
                "description": f"Column {col_name}"
            })

        logger.info("Detected %d columns from %s table", len(actual_columns), table_name)
        return {
            "schema_version": "1.0",
            "table_name": table_name,
            "description": "Dynamic schema from uploaded data",
            "columns": actual_columns
        }
    except Exception as e:
        logger.exception("Error getting actual database schema: %s", e)
        return None


def extract_sql_from_response(text: str) -> str:
    """Extract SQL statement from LLM response"""
    try:
        text = text.strip()
        # Look for JSON structure first
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and 'sql' in obj:
                    return obj['sql'].strip()
                elif isinstance(obj, dict) and 'CORRECTED_QUERY' in obj:
                    return obj['CORRECTED_QUERY'].strip()
            except:
                pass

        # Look for SQL block format
        m = re.search(r"```sql\n(.*?)```", text, re.S | re.I)
        if m:
            return m.group(1).strip()

        # Extract SELECT statement
        m = re.search(r"(SELECT[\s\S]*?);?\s*$", text, re.I)
        if m:
            return m.group(1).strip()

        return ""
    except Exception as e:
        logger.exception("Error extracting SQL: %s", e)
        return ""


def generate_sql_from_question(user_question, schema, conversation_memory=None):
    """Use LLM to generate SQL from natural language question"""
    try:
        logger.info("Generating SQL from user question: %s", user_question)

        memory_context = ""
        if conversation_memory and conversation_memory.get("history"):
            recent = conversation_memory["history"][-3:]
            memory_lines = []
            for entry in recent:
                memory_lines.append(f"Q: {entry.get('user_question', '')}")
                if entry.get('sql_query'):
                    memory_lines.append(f"SQL: {entry['sql_query']}")
                if entry.get('record_count') is not None:
                    memory_lines.append(f"Result: {entry['record_count']} rows")
            memory_context = "\n".join(memory_lines)

        prompt = (
            "You are an assistant that converts a natural language request into a single SQLite-compatible SQL query.\n"
            "All the dates are in the format YYYY-MM-DD.\n"
            "Return only valid JSON with a single key `sql` whose value is the SQL string.\n"
            "for date relates question Use Open_date, Close_date column and formulate sql like strftime('%Y', Open_date) = '2024'\n"
            "Do NOT include explanations or additional fields. Ensure the SQL is compatible with SQLite.\n\n"
            "Database schema (JSON):\n" + json.dumps(schema['columns']) + "\n\n"
            "Table name: " + schema['table_name'] + "\n"
        )

        if memory_context:
            prompt += f"\nConversation context:\n{memory_context}\n"

        prompt += (
            "User request: \"" + user_question + "\"\n\n"
            "If the request cannot be represented as a single SQL SELECT query and do not include ';' at the end of the query, return an empty string for `sql`."
        )

        response = call_llm(prompt)
        sql_query = extract_sql_from_response(response)
        logger.info("SQL generated | length=%d | query=%s", len(sql_query), sql_query[:100] if sql_query else "EMPTY")
        return sql_query
    except Exception as e:
        logger.exception("Error generating SQL from question: %s", e)
        return ""


def validate_sql_with_judge(sql_query, user_question, schema):
    """Use an LLM as a judge to validate if the generated SQL is correct"""
    sql_upper = sql_query.strip().upper()

    # Block writing operations
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
        columns_desc = "\n".join([
            f"- {col['name']} ({col['type']}): {col['description']}"
            for col in schema['columns']
        ])

        prompt = f"""You are an expert SQLite3 database validator. Your task is to judge whether the following SQL query is correct and valid.

DATABASE SCHEMA:
Table: {schema['table_name']}
Columns:
{columns_desc}

USER QUESTION: {user_question}

SQL QUERY TO VALIDATE:
{sql_query}

VALIDATION RULES:
1. Check if the SQL syntax is valid SQLite3
2. Check if column names are exact matches from the schema (no typos or incorrect column names)
3. Check if the query answers the user's question
4. Check if the date formats are correct (YYYY-MM-DD) if dates are involved.
5. Table name must be '{schema['table_name']}'
6. Do not include ';' at the end of the query
7. Reject any queries that attempt to modify data

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT * FROM {schema['table_name']} WHERE ..."}}

Do not include any explanations, text, or additional content outside the JSON object."""

        response = call_llm(prompt)
        logger.info("SQL Validation Response:\n%s", response)

        is_valid = False
        explanation = "Validation response received"
        query = None

        response_text = response.strip()

        if 'YES' in response_text.upper():
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
        pattern = r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'"])([^\2]*?)\2'

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
        st.error(f"SQL Error: {str(e)}")
        return None


def should_generate_insights(user_query, result_df):
    """Determine if a visual chart should be generated from user query"""
    if result_df is None or result_df.empty or len(result_df) < 2:
        return False
    keywords = {
        "insight", "insights", "plot", "plots", "chart", "charts",
        "graph", "graphs", "visual", "visualize", "visualise",
        "visualization", "distribution", "trend", "breakdown",
        "compare", "comparison", "percentage", "share", "dashboard",
    }
    return any(kw in user_query.lower() for kw in keywords)


def detect_chart_type(user_query):
    """Detect appropriate visualization type"""
    query_lower = user_query.lower()
    if any(k in query_lower for k in ["pie", "donut", "ratio", "proportion"]):
        return "pie"
    if any(k in query_lower for k in ["bar", "histogram", "column", "compare"]):
        return "bar"
    if any(k in query_lower for k in ["trend", "line", "growth", "over time"]):
        return "trend"
    return "auto"


def _load_and_clean_csv(uploaded_file):
    """Read CSV, strip whitespace from columns, clean rows"""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != '']
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        logger.info("Loaded and cleaned CSV | shape=(%d, %d)", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.exception("Error processing uploaded CSV: %s", e)
        return None


# =============================================================================
# CONVERSATION MEMORY FUNCTIONS
# =============================================================================

def _initialize_conversation_memory():
    """Initialize conversation memory structure"""
    return {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
    }


def _append_conversation_memory(user_query, sql_query, records, validation_result=None, answer=None):
    """Append structured conversation memory"""
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory

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

        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query

        if not answer:
            memory["last_assistant_response"] = f"SQL: {sql_query} | Records: {len(records) if records else 0}"
        else:
            memory["last_assistant_response"] = answer

        # Keep only last 10 conversations
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]

    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)


# =============================================================================
# DYNAMIC METRICS & CHARTS RETRIEVER FOR DASHBOARD TABS (CRASH-PROOF FALLBACKS)
# =============================================================================

def get_dashboard_metrics(state_filter=None, chapter_filter=None, status_filter=None, 
                        prose_filter=None, asset_filter=None, consumer_type_filter=None,
                        date_start=None, date_end=None):
    """Query dynamic metrics for transactional dashboard tab with robust schema fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            conn.close()
            return None

        # Introspect available database columns
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Total count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
        total_cases = cursor.fetchone()[0]

        # Active cases count
        active_cases = 0
        status_col = None
        if "status" in columns:
            status_col = "status"
        elif "Active_Status" in columns:
            status_col = "Active_Status"

        if status_col:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE LOWER({status_col}) IN ('active', 'open', 'y', '1') AND {where_clause}")
            active_cases = cursor.fetchone()[0]
        else:
            active_cases = total_cases

        # Converted cases count
        converted_cases = 0
        if "chapter" in columns and "original_chapter" in columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE chapter != original_chapter AND original_chapter IS NOT NULL AND original_chapter != '' AND {where_clause}")
            converted_cases = cursor.fetchone()[0]

        # Pro Se cases count
        pro_se_cases = 0
        if "prose_indicator" in columns:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE (prose_indicator = 1 OR LOWER(prose_indicator) IN ('true', '1', 'y')) AND {where_clause}")
            pro_se_cases = cursor.fetchone()[0]

        # Average Match Score
        avg_score = 0.0
        if "match_score" in columns:
            cursor.execute(f"SELECT AVG(match_score) FROM {table_name} WHERE {where_clause}")
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
        return None


def get_chart_data(state_filter=None, chapter_filter=None, status_filter=None, 
                  prose_filter=None, asset_filter=None, consumer_type_filter=None,
                  date_start=None, date_end=None):
    """Query distribution data for dashboard visual widgets with dynamic fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")

        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        base_where = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_chapter = pd.DataFrame()
        if "chapter" in columns:
            df_chapter = pd.read_sql_query(
                f"SELECT chapter as Chapter, COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND chapter IS NOT NULL AND chapter != '' GROUP BY chapter", conn
            )

        df_client = pd.DataFrame()
        client_col = None
        for col in ["record_type", "consumer_type", "notice_type", "client"]:
            if col in columns:
                client_col = col
                break
        if client_col:
            df_client = pd.read_sql_query(
                f"SELECT {client_col} as [Record Type], COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND {client_col} IS NOT NULL AND {client_col} != '' GROUP BY {client_col}", conn
            )

        df_state = pd.DataFrame()
        if "State" in columns:
            df_state = pd.read_sql_query(
                f"SELECT State, COUNT(*) as Count FROM {table_name} WHERE ({base_where}) AND State IS NOT NULL AND State != '' GROUP BY State ORDER BY Count DESC LIMIT 10", conn
            )

        df_trend = pd.DataFrame()
        date_col = None
        if "Open_date" in columns:
            date_col = "Open_date"
        elif "date_filed" in columns:
            date_col = "date_filed"
        
        if date_col:
            df_trend = pd.read_sql_query(
                f"""
                SELECT substr({date_col}, 1, 4) as Year, COUNT(*) as Count
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
                                prose_filter=None, asset_filter=None, consumer_type_filter=None,
                                date_start=None, date_end=None):
    """Get case status distribution for pie chart visualization"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_status = pd.DataFrame()
        if "status" in columns:
            df_status = pd.read_sql_query(
                f"SELECT status as [Status], COUNT(*) as Count FROM {table_name} WHERE status IS NOT NULL AND status != '' AND {where_clause} GROUP BY status ORDER BY Count DESC",
                conn
            )
        else:
            # Fallback: use case disposition or chapter as status proxy
            if "chapter" in columns:
                df_status = pd.read_sql_query(
                    f"SELECT chapter as [Status], COUNT(*) as Count FROM {table_name} WHERE chapter IS NOT NULL AND chapter != '' AND {where_clause} GROUP BY chapter ORDER BY Count DESC",
                    conn
                )
        
        conn.close()
        return df_status
    except Exception as e:
        logger.error(f"Error getting case status distribution: {e}")
        return pd.DataFrame()


# =============================================================================
# BANKRUPTCY DATA INTELLIGENCE MODULES (FALLBACK PROTECTION)
# =============================================================================

def get_advanced_chapter_conversion_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                                            prose_filter=None, asset_filter=None, consumer_type_filter=None,
                                            date_start=None, date_end=None):
    """Advanced chapter analysis with risk metrics, conversion flows, and distribution insights"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Chapter distribution with risk metrics
        df_chapter_risk = pd.DataFrame()
        if "chapter" in columns:
            if "match_score" in columns:
                df_chapter_risk = pd.read_sql_query(
                    f"""
                    SELECT 
                        chapter as [Chapter],
                        COUNT(*) as [Total Cases],
                        ROUND(AVG(match_score), 1) as [Avg Risk Score],
                        ROUND(MIN(match_score), 1) as [Min Risk],
                        ROUND(MAX(match_score), 1) as [Max Risk],
                        SUM(CASE WHEN match_score >= 80 THEN 1 ELSE 0 END) as [High Risk Cases]
                    FROM {table_name}
                    WHERE chapter IS NOT NULL AND chapter != '' AND {where_clause}
                    GROUP BY chapter
                    ORDER BY [Total Cases] DESC
                    """, conn
                )
            else:
                df_chapter_risk = pd.read_sql_query(
                    f"""
                    SELECT 
                        chapter as [Chapter],
                        COUNT(*) as [Total Cases],
                        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE {where_clause}), 1) as [% of Portfolio]
                    FROM {table_name}
                    WHERE chapter IS NOT NULL AND chapter != '' AND {where_clause}
                    GROUP BY chapter
                    ORDER BY [Total Cases] DESC
                    """, conn
                )

        # Conversion paths if columns exist
        df_conversions = pd.DataFrame()
        if "chapter" in columns and "original_chapter" in columns:
            df_conversions = pd.read_sql_query(
                f"""
                SELECT 
                    original_chapter as [From], 
                    chapter as [To], 
                    COUNT(*) as [Conversions],
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE original_chapter != chapter AND original_chapter IS NOT NULL AND {where_clause}), 1) as [% of Conversions]
                FROM {table_name} 
                WHERE original_chapter != chapter AND original_chapter IS NOT NULL AND original_chapter != '' AND chapter IS NOT NULL AND chapter != '' AND {where_clause}
                GROUP BY original_chapter, chapter 
                ORDER BY [Conversions] DESC
                LIMIT 5
                """, conn
            )

        # Pro Se and Asset distribution by chapter
        df_chapter_demographics = pd.DataFrame()
        if "chapter" in columns and "prose_indicator" in columns:
            df_chapter_demographics = pd.read_sql_query(
                f"""
                SELECT 
                    chapter as [Chapter],
                    SUM(CASE WHEN prose_indicator IN ('Y', '1', 'true') THEN 1 ELSE 0 END) as [Pro Se],
                    SUM(CASE WHEN prose_indicator NOT IN ('Y', '1', 'true') THEN 1 ELSE 0 END) as [Represented]
                FROM {table_name}
                WHERE chapter IS NOT NULL AND chapter != '' AND {where_clause}
                GROUP BY chapter
                ORDER BY chapter
                """, conn
            )

        conn.close()
        return df_chapter_risk, df_conversions, df_chapter_demographics
    except Exception as e:
        logger.error(f"Error in advanced chapter conversion insights: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()



def get_legal_representation_insights(state_filter=None, chapter_filter=None, status_filter=None, 
                                    prose_filter=None, asset_filter=None, consumer_type_filter=None,
                                    date_start=None, date_end=None):
    """Query representation data, pro se rates, top trustees and attorneys with dynamic check fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_rep = pd.DataFrame()
        if "prose_indicator" in columns:
            df_rep = pd.read_sql_query(
                f"""
                SELECT CASE WHEN prose_indicator = 1 OR LOWER(prose_indicator) IN ('true', '1', 'y') THEN 'Pro Se (Self-Represented)' ELSE 'Represented by Counsel' END as [Counsel Status],
                       COUNT(*) as [Case Count]
                FROM {table_name}
                WHERE {where_clause}
                GROUP BY [Counsel Status]
                """, conn
            )

        df_attorneys = pd.DataFrame()
        attorney_cols = [c for c in ["Attorny_DBA", "Attorny_First_Name", "Attorny_lastt_Name"] if c in columns]
        if len(attorney_cols) > 0:
            name_expr = "COALESCE(Attorny_DBA, '')"
            if "Attorny_First_Name" in columns and "Attorny_lastt_Name" in columns:
                name_expr = "COALESCE(NULLIF(Attorny_DBA, ''), Attorny_First_Name || ' ' || Attorny_lastt_Name)"
            elif "Attorny_lastt_Name" in columns:
                name_expr = "COALESCE(NULLIF(Attorny_DBA, ''), Attorny_lastt_Name)"

            df_attorneys = pd.read_sql_query(
                f"""
                SELECT {name_expr} as [Firm / Attorney Name], COUNT(*) as [Cases Handled]
                FROM {table_name}
                WHERE [Firm / Attorney Name] IS NOT NULL AND [Firm / Attorney Name] != '' AND [Firm / Attorney Name] != ' ' AND {where_clause}
                GROUP BY [Firm / Attorney Name]
                ORDER BY [Cases Handled] DESC
                LIMIT 10
                """, conn
            )

        df_trustees = pd.DataFrame()
        if "trustee_name" in columns:
            df_trustees = pd.read_sql_query(
                f"""
                SELECT trustee_name as [Trustee Name], COUNT(*) as [Cases Administered]
                FROM {table_name}
                WHERE trustee_name IS NOT NULL AND trustee_name != '' AND {where_clause}
                GROUP BY trustee_name
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

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Top clients by case count with risk metrics
        df_top_clients = pd.DataFrame()
        if "client" in columns:
            if "match_score" in columns:
                df_top_clients = pd.read_sql_query(
                    f"""
                    SELECT 
                        client as [Client],
                        COUNT(*) as [Case Count],
                        ROUND(AVG(match_score), 1) as [Avg Risk Score],
                        SUM(CASE WHEN match_score >= 80 THEN 1 ELSE 0 END) as [High Risk]
                    FROM {table_name}
                    WHERE client IS NOT NULL AND client != '' AND client != ' ' AND {where_clause}
                    GROUP BY client
                    ORDER BY [Case Count] DESC
                    LIMIT 8
                    """, conn
                )
            else:
                df_top_clients = pd.read_sql_query(
                    f"""
                    SELECT 
                        client as [Client],
                        COUNT(*) as [Case Count]
                    FROM {table_name}
                    WHERE client IS NOT NULL AND client != '' AND client != ' ' AND {where_clause}
                    GROUP BY client
                    ORDER BY [Case Count] DESC
                    LIMIT 8
                    """, conn
                )

        # Client distribution by chapter
        df_client_chapter = pd.DataFrame()
        if "client" in columns and "chapter" in columns:
            df_client_chapter = pd.read_sql_query(
                f"""
                SELECT 
                    client as [Client],
                    chapter as [Chapter],
                    COUNT(*) as [Cases]
                FROM {table_name}
                WHERE client IS NOT NULL AND client != '' AND chapter IS NOT NULL AND {where_clause}
                GROUP BY client, chapter
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
                              prose_filter=None, asset_filter=None, consumer_type_filter=None,
                              date_start=None, date_end=None):
    """Query asset indicators and geographical trends with dynamic column check fallbacks"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build WHERE clause from filters
        where_clauses = []
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        df_assets = pd.DataFrame()
        if "Asset_indicator" in columns:
            df_assets = pd.read_sql_query(
                f"""
                SELECT CASE WHEN Asset_indicator = 1 OR LOWER(Asset_indicator) IN ('true', '1', 'y') THEN 'Asset Cases (Assets Present)' ELSE 'No-Asset Cases' END as [Liquidation Type],
                       COUNT(*) as [Case Count]
                FROM {table_name}
                WHERE {where_clause}
                GROUP BY [Liquidation Type]
                """, conn
            )

        df_geo = pd.DataFrame()
        if "City" in columns and "State" in columns:
            df_geo = pd.read_sql_query(
                f"""
                SELECT City || ', ' || State as [Jurisdiction], COUNT(*) as [Filings Density]
                FROM {table_name}
                WHERE City IS NOT NULL AND City != '' AND State IS NOT NULL AND State != '' AND {where_clause}
                GROUP BY [Jurisdiction]
                ORDER BY [Filings Density] DESC
                LIMIT 10
                """, conn
            )
        elif "State" in columns:
            df_geo = pd.read_sql_query(
                f"""
                SELECT State as [Jurisdiction], COUNT(*) as [Filings Density]
                FROM {table_name}
                WHERE State IS NOT NULL AND State != '' AND {where_clause}
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

        df_scores = pd.DataFrame()
        if "match_score" in columns:
            df_scores = pd.read_sql_query(
                f"SELECT match_score as [Match Score], COUNT(*) as Count FROM {table_name} WHERE match_score IS NOT NULL GROUP BY match_score ORDER BY match_score", conn
            )

        df_high_risk = pd.DataFrame()
        if "Ac_no" in columns:
            name_expr = "Ac_no"
            if "First_name" in columns and "Last_name" in columns:
                name_expr = "First_name || ' ' || Last_name"
            elif "PD_First_Name" in columns and "PD_lastt_Name" in columns:
                name_expr = "PD_First_Name || ' ' || PD_lastt_Name"

            state_col = "State" if "State" in columns else ("PD_State" if "PD_State" in columns else "NULL")
            chapter_col = "chapter" if "chapter" in columns else "NULL"
            score_col = "match_score" if "match_score" in columns else "100"
            status_col = "status" if "status" in columns else "NULL"

            df_high_risk = pd.read_sql_query(
                f"""
                SELECT Ac_no as [Account No], {name_expr} as Name,
                       {state_col} as State, {chapter_col} as Chapter, {score_col} as [Match Score], {status_col} as Status
                FROM {table_name}
                WHERE {score_col} >= 95
                ORDER BY {score_col} DESC, Ac_no ASC
                """, conn
            )
        conn.close()
        return df_scores, df_high_risk
    except Exception as e:
        logger.error(f"Error querying predictive data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
#Forecasting IMPROVEMENTS: TIME SERIES FORECASTING & EARLY WARNINGS
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
        
        # Detect channel/source columns
        channel_cols = [c for c in ["channel", "source", "record_type", "consumer_type", "notice_type"] if c in columns]
        date_col = None
        if "Open_date" in columns:
            date_col = "Open_date"
        elif "date_filed" in columns:
            date_col = "date_filed"
        
        if channel_cols and date_col:
            channel_col = channel_cols[0]
            df_forecast = pd.read_sql_query(
                f"""
                SELECT {channel_col} as Channel, 
                       substr({date_col}, 1, 4) as Year, 
                       COUNT(*) as [Filing Count],
                       AVG(CAST(match_score AS FLOAT)) as [Avg Risk Score]
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
        
        # Detect source/client columns
        source_cols = [c for c in ["source", "client", "originating_source", "filing_source"] if c in columns]
        date_col = None
        if "Open_date" in columns:
            date_col = "Open_date"
        elif "date_filed" in columns:
            date_col = "date_filed"
        
        if source_cols and date_col:
            source_col = source_cols[0]
            df_forecast = pd.read_sql_query(
                f"""
                SELECT {source_col} as Source, 
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
    """Early warning radar system with analysis filter (data range)"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Determine date column
        date_col = None
        if "Open_date" in columns:
            date_col = "Open_date"
        elif "date_filed" in columns:
            date_col = "date_filed"

        alerts_data = {}
        
        # High-risk cluster detection
        if "match_score" in columns:
            risk_threshold = 80
            date_filter = ""
            if date_col and date_range_start and date_range_end:
                date_filter = f" AND {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_high_risk_clusters = pd.read_sql_query(
                f"""
                SELECT COUNT(*) as [High Risk Cases], 
                       ROUND(AVG(match_score), 2) as [Avg Risk Level],
                       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE {date_col} IS NOT NULL AND {date_col} != '' {date_filter}) as [% of Total]
                FROM {table_name}
                WHERE match_score >= {risk_threshold} AND {date_col} IS NOT NULL AND {date_col} != '' {date_filter}
                """, conn
            )
            alerts_data['high_risk'] = df_high_risk_clusters
        
        # Sudden spike detection (anomaly detection)
        if date_col:
            date_filter = ""
            if date_range_start and date_range_end:
                date_filter = f" WHERE {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_spike_detection = pd.read_sql_query(
                f"""
                SELECT substr({date_col}, 1, 7) as Month, 
                       COUNT(*) as [Monthly Filings],
                       LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7)) as [Previous Month],
                       ROUND((COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7))) * 100.0 / 
                             NULLIF(LAG(COUNT(*)) OVER (ORDER BY substr({date_col}, 1, 7)), 0), 2) as [% Change]
                FROM {table_name}
                WHERE {date_col} IS NOT NULL AND {date_col} != '' {date_filter}
                GROUP BY Month
                ORDER BY Month DESC
                LIMIT 12
                """, conn
            )
            alerts_data['spike_detection'] = df_spike_detection
        
        # Pattern anomalies
        if "chapter" in columns and date_col:
            date_filter = ""
            if date_range_start and date_range_end:
                date_filter = f" AND {date_col} BETWEEN '{date_range_start}' AND '{date_range_end}'"
            
            df_chapter_anomalies = pd.read_sql_query(
                f"""
                SELECT chapter, COUNT(*) as [Case Count],
                       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name} WHERE chapter IS NOT NULL AND chapter != '' {date_filter}), 2) as [% Distribution]
                FROM {table_name}
                WHERE chapter IS NOT NULL AND chapter != '' {date_filter}
                GROUP BY chapter
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
# ADVANCED DRILL-DOWN & FILTERING FUNCTIONS FORAnalytics
# =============================================================================

def get_filtered_cases(state_filter=None, chapter_filter=None, status_filter=None, 
                       prose_filter=None, asset_filter=None, consumer_type_filter=None,
                       date_start=None, date_end=None, limit=500):
    """Retrieve filtered case details for drill-down analysis"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        # Build base query
        name_expr = "Ac_no"
        if "First_name" in columns and "Last_name" in columns:
            name_expr = "First_name || ' ' || Last_name"
        elif "PD_First_Name" in columns and "PD_lastt_Name" in columns:
            name_expr = "PD_First_Name || ' ' || PD_lastt_Name"

        where_clauses = []
        
        # Add filters
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        
        if prose_filter and "prose_indicator" in columns:
            where_clauses.append(f"prose_indicator = '{prose_filter}'")
        
        if asset_filter and "Asset_indicator" in columns:
            where_clauses.append(f"Asset_indicator = '{asset_filter}'")
        
        if consumer_type_filter and "consumer_type" in columns:
            where_clauses.append(f"consumer_type = '{consumer_type_filter}'")
        
        if date_start and date_end and "Open_date" in columns:
            where_clauses.append(f"Open_date BETWEEN '{date_start}' AND '{date_end}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        SELECT Ac_no as [Case #], {name_expr} as Name, 
               COALESCE(State, 'N/A') as State, COALESCE(chapter, 'N/A') as Chapter,
               COALESCE(status, 'N/A') as Status, Open_date as [Open Date],
               COALESCE(match_score, 0) as [Risk Score],
               CASE WHEN prose_indicator IN ('Y', '1', 'true') THEN 'Pro Se' ELSE 'Represented' END as [Representation],
               CASE WHEN Asset_indicator IN ('Y', '1', 'true') THEN 'Asset' ELSE 'No Asset' END as [Asset Status],
               COALESCE(consumer_type, 'N/A') as [Consumer Type]
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY match_score DESC, Open_date DESC
        LIMIT {limit}
        """
        
        df_filtered = pd.read_sql_query(query, conn)
        conn.close()
        return df_filtered
    except Exception as e:
        logger.error(f"Error filtering cases: {e}")
        return pd.DataFrame()


def get_case_detail(case_number):
    """Get detailed information for a specific case"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        
        query = f"SELECT * FROM {table_name} WHERE Ac_no = {case_number} LIMIT 1"
        df_detail = pd.read_sql_query(query, conn)
        conn.close()
        
        return df_detail
    except Exception as e:
        logger.error(f"Error retrieving case detail: {e}")
        return pd.DataFrame()


def get_drilldown_summary_stats(state_filter=None, chapter_filter=None, status_filter=None):
    """Get summary statistics for filtered data"""
    try:
        table_name = get_db_table_name()
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = {row[1] for row in cursor.fetchall()}

        where_clauses = []
        
        if state_filter and "State" in columns:
            where_clauses.append(f"State = '{state_filter}'")
        
        if chapter_filter and "chapter" in columns:
            where_clauses.append(f"chapter = {chapter_filter}")
        
        if status_filter and "status" in columns:
            where_clauses.append(f"status = '{status_filter}'")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Count query
        count_query = f"SELECT COUNT(*) as total FROM {table_name} WHERE {where_clause}"
        total = pd.read_sql_query(count_query, conn).iloc[0]['total']
        
        # Average match score
        avg_score = 0
        if "match_score" in columns:
            score_query = f"SELECT ROUND(AVG(match_score), 1) as avg_score FROM {table_name} WHERE {where_clause}"
            avg_score = pd.read_sql_query(score_query, conn).iloc[0]['avg_score'] or 0
        
        # Pro Se count
        pro_se_count = 0
        if "prose_indicator" in columns:
            prose_query = f"SELECT COUNT(*) as pro_se FROM {table_name} WHERE {where_clause} AND prose_indicator IN ('Y', '1', 'true')"
            pro_se_count = pd.read_sql_query(prose_query, conn).iloc[0]['pro_se']
        
        # Asset count
        asset_count = 0
        if "Asset_indicator" in columns:
            asset_query = f"SELECT COUNT(*) as assets FROM {table_name} WHERE {where_clause} AND Asset_indicator IN ('Y', '1', 'true')"
            asset_count = pd.read_sql_query(asset_query, conn).iloc[0]['assets']
        
        conn.close()
        
        return {
            'total': total,
            'avg_score': avg_score,
            'pro_se': pro_se_count,
            'assets': asset_count
        }
    except Exception as e:
        logger.error(f"Error getting drilldown stats: {e}")
        return {'total': 0, 'avg_score': 0, 'pro_se': 0, 'assets': 0}


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    logger.info("=== Session run initiated ===")

    # Initialize states
    if "data_in_db" not in st.session_state:
        try:
            conn = sqlite3.connect("data.db")
            cursor = conn.cursor()
            table_name = get_db_table_name()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            st.session_state.data_in_db = cursor.fetchone() is not None
            conn.close()
        except Exception:
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
                "content": " **Welcome to Bankruptcy GenBI Assistant!** Ask me any natural language questions about the loaded filings data (e.g., *'Show cases filed in 2021'* or *'Plot a pie chart of cases chapter breakdown'*)."
            }
        ]

    # Load default schema
    schema = load_schema()
    if not schema:
        st.error("Fatal Error: Could not read schema configuration. Make sure schema.json is in the workspace.")
        return

    # Dynamic schema fallback
    active_schema = st.session_state.actual_schema if st.session_state.actual_schema else schema

    # =========================================================================
    # SIDEBAR NAVIGATION (Light Gray Layout Theme)
    # =========================================================================
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1rem 0; text-align: center;">
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                     GenBI 
                </span>
                <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.2rem; font-weight: 500; margin-bottom: 0px;">Bankruptcy Analytics Portal</p>
            </div>
            <hr style="margin: 0.5rem 0 1.5rem 0 !important; border-top: 1px solid #cbd5e1;">
            """,
            unsafe_allow_html=True
        )
        selected_tab = st.radio(
            "Select Workspace",
            ["Data", "Analytics", "Forecasting", "AI Assistant"],
            label_visibility="collapsed"
        )

    # Header section
    st.markdown('<div class="main-title"> Bankruptcy GenBI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enterprise Bankruptcy & Risk Intelligence Dashboard</div>', unsafe_allow_html=True)

    # =========================================================================
    # DATA WORKSPACE TABS SWITCH
    # =========================================================================

    # -----------------------------------------------------------------
    # TAB 0:Data (Separated Panel)
    # -----------------------------------------------------------------
    if selected_tab == "Data":
        st.subheader("Data & Repository Management")
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

                        st.success(f"✅ Loaded: {uploaded_file.name}")
                        st.rerun()
                    else:
                        st.error("Failed to parse file. Ensure format is standard CSV.")

        # Display internal database status parameters
        if st.session_state.data_in_db:
            try:
                conn = sqlite3.connect("data.db")
                table_name = get_db_table_name()
                row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn)['count'].values[0]

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.2rem; border-radius: 12px; border: 1px solid #bbf7d0; margin-top: 1rem; margin-bottom: 1rem;">
                        <span style="color: #15803d; font-weight: 700; font-size: 1.25rem;">✨ Database Active</span>
                        <div style="color: #166534; font-size: 0.9rem; font-weight: 600; margin-top: 0.4rem;">
                            <strong>{row_count:,}</strong> records active in sqlite table: <code>{table_name}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Optional Preview panel
                with st.expander(" View Data Preview", expanded=True):
                    df_preview = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 50", conn)
                    st.dataframe(df_preview, width="stretch", hide_index=True)

                # Dynamic Columns dictionary list
                if st.session_state.actual_schema:
                    cols_list = [c['name'] for c in st.session_state.actual_schema['columns']]
                    with st.expander(f" Columns Dictionary ({len(cols_list)})", expanded=False):
                        for col in cols_list:
                            st.caption(f"• {col}")

                conn.close()
            except Exception as e:
                logger.error(f"Error drawing data management cards: {e}")
                st.session_state.data_in_db = False
        else:
            st.info("No active table detected. Please upload a CSV to populate database.")

    # -----------------------------------------------------------------
    # TAB 1:Analytics
    # -----------------------------------------------------------------
    elif selected_tab == "Analytics":
        st.subheader("Transactional Analytics & Portfolios")
        st.markdown("Dynamic case-profiling intelligence derived from matching scores, jurisdiction networks, and representation distributions.")

        if not st.session_state.data_in_db:
            st.warning(" No active dataset found. Please upload a CSV on the 'Data Ingestion' workspace to begin.")
        else:
            metrics = get_dashboard_metrics()
            if metrics:
                # Top metrics banner
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Case Filings", f"{metrics['total']:,}")
                c2.metric("Active Case Count", f"{metrics['active']:,}")
                c3.metric("Converted Chapters", f"{metrics['converted']:,}", "Original vs Current chapter mismatch")
                c4.metric("Pro Se Debtor Rate", f"{metrics['pro_se']:,} cases", "Self-represented files")

                st.markdown("<br>", unsafe_allow_html=True)

                # Advanced Drill-Down Filters (Integrated)
                with st.expander(" Advanced Drill-Down Filters & Case Analysis", expanded=False):
                    st.markdown("Filter cases by multiple criteria to discover insights and analyze specific case cohorts.")
                    
                    # Filter controls
                    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                    
                    with filter_col1:
                        state_options = ["All States"] + sorted(list(set(["TX", "CA", "NY", "IL", "AZ", "FL", "PA", "OH", "GA", "NC"])))
                        selected_state = st.selectbox("State", state_options, key="drilldown_state")
                        state_filter = selected_state if selected_state != "All States" else None
                    
                    with filter_col2:
                        chapter_options = ["All Chapters", 7, 11, 13]
                        selected_chapter = st.selectbox("Chapter", chapter_options, key="drilldown_chapter")
                        chapter_filter = selected_chapter if selected_chapter != "All Chapters" else None
                    
                    with filter_col3:
                        status_options = ["All Status"] + ["Active", "Closed", "Pending", "Converted", "Dismissed", "Discharged"]
                        selected_status = st.selectbox("Status", status_options, key="drilldown_status")
                        status_filter = selected_status if selected_status != "All Status" else None
                    
                    with filter_col4:
                        consumer_options = ["All Types", "Individual", "Corporate", "SME", "Business", "Partnership", "Trust"]
                        selected_consumer = st.selectbox("Consumer Type", consumer_options, key="drilldown_consumer")
                        consumer_filter = selected_consumer if selected_consumer != "All Types" else None
                    
                    # Secondary filters
                    sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
                    
                    with sec_col1:
                        prose_options = ["All", "Pro Se Only", "Represented Only"]
                        prose_selected = st.selectbox("Representation", prose_options, key="drilldown_prose")
                        prose_filter = "Y" if prose_selected == "Pro Se Only" else ("N" if prose_selected == "Represented Only" else None)
                    
                    with sec_col2:
                        asset_options = ["All", "Asset Cases Only", "No-Asset Cases Only"]
                        asset_selected = st.selectbox("Asset Status", asset_options, key="drilldown_asset")
                        asset_filter = "Y" if asset_selected == "Asset Cases Only" else ("N" if asset_selected == "No-Asset Cases Only" else None)
                    
                    with sec_col3:
                        date_start = st.date_input("Date From", value=None, key="drilldown_date_start")
                    
                    with sec_col4:
                        date_end = st.date_input("Date To", value=None, key="drilldown_date_end")
                    
                    # Get filtered data
                    date_start_str = date_start.strftime('%Y-%m-%d') if date_start else None
                    date_end_str = date_end.strftime('%Y-%m-%d') if date_end else None
                    
                    df_filtered = get_filtered_cases(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        asset_filter=asset_filter,
                        consumer_type_filter=consumer_filter,
                        date_start=date_start_str,
                        date_end=date_end_str
                    )
                    
                    # Summary statistics
                    if not df_filtered.empty:
                        st.markdown("---")
                        st.markdown("#### Summary Statistics for Filtered Data")
                        
                        summary_stats = get_drilldown_summary_stats(state_filter, chapter_filter, status_filter)
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        stat_col1.metric("Total Cases", f"{summary_stats['total']:,}")
                        stat_col2.metric("Avg Risk Score", f"{summary_stats['avg_score']}")
                        stat_col3.metric("Pro Se Cases", f"{summary_stats['pro_se']:,}")
                        stat_col4.metric("Asset Cases", f"{summary_stats['assets']:,}")
                        
                        st.markdown("---")
                        st.markdown("#### Detailed Case List")
                        
                        # Display filtered cases with expansion for details
                        st.dataframe(df_filtered, width="stretch", hide_index=True, use_container_width=True)
                        
                        # Download option
                        csv = df_filtered.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Cases as CSV",
                            data=csv,
                            file_name=f"filtered_cases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No cases found matching the selected filters. Try adjusting the criteria.")

                st.markdown("<br>", unsafe_allow_html=True)

                # 3 Main tabs
                tab_overview, tab_dynamics, tab_legal, tab_assets = st.tabs([
                    " Overview Metrics",
                    " Case & Chapter Dynamics",
                    " Representation & Professional Networks",
                    " Geographic & Asset Profile Hotspots"
                ])

                # Tab 0: Overview Metrics
                with tab_overview:
                    st.markdown("### Portfolio Summary Statistics")
                    
                    overview_col1, overview_col2 = st.columns(2)
                    with overview_col1:
                        st.markdown("#### Case Status Distribution")
                        df_status = get_case_status_distribution(
                            state_filter=state_filter,
                            chapter_filter=chapter_filter,
                            status_filter=status_filter,
                            prose_filter=prose_filter,
                            asset_filter=asset_filter,
                            consumer_type_filter=consumer_filter,
                            date_start=date_start_str,
                            date_end=date_end_str
                        )
                        
                        if df_status is not None and not df_status.empty:
                            # Create pie chart
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=df_status['Status'],
                                values=df_status['Count'],
                                hole=0,
                                marker=dict(
                                    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                                    line=dict(color='white', width=2)
                                ),
                                textposition='inside',
                                textinfo='label+percent',
                                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                            )])
                            
                            fig_pie.update_layout(
                                height=400,
                                showlegend=True,
                                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                                margin=dict(l=0, r=100, t=20, b=0)
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("No status data available for visualization")
                        
                    with overview_col2:
                        st.markdown("#### Filing Rate Trends")
                        df_chapter_old, df_client, df_state_old, df_trend = get_chart_data(
                            state_filter=state_filter,
                            chapter_filter=chapter_filter,
                            status_filter=status_filter,
                            prose_filter=prose_filter,
                            asset_filter=asset_filter,
                            consumer_type_filter=consumer_filter,
                            date_start=date_start_str,
                            date_end=date_end_str
                        )
                        if df_trend is not None and not df_trend.empty:
                            df_trend_chart = df_trend.set_index("Year")
                            st.line_chart(df_trend_chart, width="stretch")
                        else:
                            st.caption("No trend data available")

                # Quadrant 1: Case & Chapter Dynamics
                with tab_dynamics:
                    st.markdown("### Case Chapter Distributions & Conversions")
                    df_chapter_risk, df_conversions, df_chapter_demo = get_advanced_chapter_conversion_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        asset_filter=asset_filter,
                        consumer_type_filter=consumer_filter,
                        date_start=date_start_str,
                        date_end=date_end_str
                    )

                    col_dyn1, col_dyn2 = st.columns([1, 1.2])
                    
                    with col_dyn1:
                        st.markdown("#### Chapter Distribution & Risk Profile")
                        if df_chapter_risk is not None and not df_chapter_risk.empty:
                            # Create horizontal bar chart with risk scores
                            fig_chapter = go.Figure()
                            
                            fig_chapter.add_trace(go.Bar(
                                y=df_chapter_risk['Chapter'].astype(str),
                                x=df_chapter_risk['Total Cases'],
                                orientation='h',
                                marker=dict(
                                    color=df_chapter_risk.get('Avg Risk Score', 0),
                                    colorscale='RdYlGn_r',
                                    showscale=True,
                                    colorbar=dict(title="Avg Risk<br>Score", thickness=15, len=0.7)
                                ),
                                text=df_chapter_risk['Total Cases'],
                                textposition='outside',
                                hovertemplate='<b>Chapter %{y}</b><br>Cases: %{x}<br>Avg Risk: %{marker.color:.1f}<extra></extra>'
                            ))
                            
                            fig_chapter.update_layout(
                                height=350,
                                xaxis_title="Number of Cases",
                                yaxis_title="Chapter",
                                margin=dict(l=50, r=150, t=20, b=40),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_chapter, use_container_width=True)
                        else:
                            st.caption("No chapter data available")

                    with col_dyn2:
                        st.markdown("#### Chapter Conversion Flows & Insights")
                        st.caption("Tracks legal progression patterns and case transformations")
                        
                        if df_conversions is not None and not df_conversions.empty:
                            # Create conversion flow visualization
                            fig_conversion = go.Figure()
                            
                            fig_conversion.add_trace(go.Bar(
                                y=(df_conversions['From'].astype(str) + ' → ' + df_conversions['To'].astype(str)),
                                x=df_conversions['Conversions'],
                                orientation='h',
                                marker=dict(
                                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(df_conversions)],
                                    line=dict(color='white', width=1)
                                ),
                                text=df_conversions['Conversions'],
                                textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Conversions: %{x}<br>%{customdata}% of total<extra></extra>',
                                customdata=df_conversions['% of Conversions']
                            ))
                            
                            fig_conversion.update_layout(
                                height=350,
                                xaxis_title="Number of Conversions",
                                margin=dict(l=80, r=50, t=20, b=40),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_conversion, use_container_width=True)
                        else:
                            st.info("No chapter conversion paths identified. Cases appear to have consistent chapter classification.")
                    
                    # Pro Se vs Represented by Chapter
                    if df_chapter_demo is not None and not df_chapter_demo.empty:
                        st.markdown("---")
                        st.markdown("#### Representation Distribution by Chapter")
                        
                        fig_rep_dist = go.Figure()
                        
                        fig_rep_dist.add_trace(go.Bar(
                            name='Pro Se',
                            x=df_chapter_demo['Chapter'],
                            y=df_chapter_demo['Pro Se'],
                            marker_color='#d62728'
                        ))
                        
                        fig_rep_dist.add_trace(go.Bar(
                            name='Represented',
                            x=df_chapter_demo['Chapter'],
                            y=df_chapter_demo['Represented'],
                            marker_color='#2ca02c'
                        ))
                        
                        fig_rep_dist.update_layout(
                            barmode='stack',
                            height=300,
                            xaxis_title="Chapter",
                            yaxis_title="Number of Cases",
                            hovermode='x unified',
                            margin=dict(l=50, r=50, t=20, b=40)
                        )
                        
                        st.plotly_chart(fig_rep_dist, use_container_width=True)

                # Quadrant 2: Representation & Professional Networks
                with tab_legal:
                    st.markdown("### Counsel Representation & Judicial Concentrations")
                    df_rep, df_attorneys, df_trustees = get_legal_representation_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        asset_filter=asset_filter,
                        consumer_type_filter=consumer_filter,
                        date_start=date_start_str,
                        date_end=date_end_str
                    )

                    col_leg1, col_leg2, col_leg3 = st.columns(3)

                    with col_leg1:
                        st.markdown("#### Representation Distribution")
                        if df_rep is not None and not df_rep.empty:
                            df_rep_chart = df_rep.set_index("Counsel Status")
                            st.bar_chart(df_rep_chart, width="stretch")
                        else:
                            st.caption("No representation attributes found.")

                    with col_leg2:
                        st.markdown("#### Top Debtor Attorneys / Law Firms")
                        if df_attorneys is not None and not df_attorneys.empty:
                            st.dataframe(df_attorneys, width="stretch", hide_index=True)
                        else:
                            st.caption("No attorney listings discovered.")

                    with col_leg3:
                        st.markdown("#### Top Clients by Case Portfolio")
                        st.caption("Leading clients ranked by bankruptcy case count and risk profile")
                        
                        df_top_clients, df_client_chapter = get_client_insights(
                            state_filter=state_filter,
                            chapter_filter=chapter_filter,
                            status_filter=status_filter,
                            prose_filter=prose_filter,
                            asset_filter=asset_filter,
                            consumer_type_filter=consumer_filter,
                            date_start=date_start_str,
                            date_end=date_end_str
                        )
                        
                        if df_top_clients is not None and not df_top_clients.empty:
                            # Create horizontal bar chart with risk scoring
                            fig_clients = go.Figure()
                            
                            fig_clients.add_trace(go.Bar(
                                y=df_top_clients['Client'],
                                x=df_top_clients['Case Count'],
                                orientation='h',
                                marker=dict(
                                    color=df_top_clients.get('Avg Risk Score', 0) if 'Avg Risk Score' in df_top_clients.columns else 50,
                                    colorscale='RdYlGn_r',
                                    showscale=True,
                                    colorbar=dict(title="Risk<br>Score", thickness=12, len=0.8)
                                ),
                                text=df_top_clients['Case Count'],
                                textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Cases: %{x}<br>Avg Risk: %{marker.color:.1f}<extra></extra>'
                            ))
                            
                            fig_clients.update_layout(
                                height=350,
                                xaxis_title="Number of Cases",
                                yaxis_title="",
                                margin=dict(l=150, r=120, t=20, b=40),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_clients, use_container_width=True)
                        else:
                            st.info("No client data available for visualization")

                # Quadrant 3: Geographic & Asset Profile Hotspots
                with tab_assets:
                    st.markdown("### Liquidation Exposure & Geographic Densities")
                    df_assets, df_geo = get_asset_and_geo_insights(
                        state_filter=state_filter,
                        chapter_filter=chapter_filter,
                        status_filter=status_filter,
                        prose_filter=prose_filter,
                        asset_filter=asset_filter,
                        consumer_type_filter=consumer_filter,
                        date_start=date_start_str,
                        date_end=date_end_str
                    )

                    col_asset1, col_asset2 = st.columns(2)

                    with col_asset1:
                        st.markdown("#### Filing Volume Breakdown by Assets Presence")
                        st.caption("Asset cases require extensive tracking as they present actual recovery value for creditors.")
                        if df_assets is not None and not df_assets.empty:
                            df_assets_chart = df_assets.set_index("Liquidation Type")
                            st.bar_chart(df_assets_chart, width="stretch")
                        else:
                            st.caption("No asset parameters present in database.")

                    with col_asset2:
                        st.markdown("#### Filing Density: Top 10 Metropolitan Districts")
                        if df_geo is not None and not df_geo.empty:
                            df_geo_chart = df_geo.set_index("Jurisdiction")
                            st.bar_chart(df_geo_chart, width="stretch")
                        else:
                            st.caption("Location parameters missing from schema.")
            else:
                st.error("Error reading dashboard intelligence metrics from SQLite.")

    # -----------------------------------------------------------------
    # TAB 2:Forecasting
    # -----------------------------------------------------------------
    elif selected_tab == "Forecasting":
        st.subheader(" Predictive Risk & Forecasting")
        st.markdown("Advanced risk scoring, time series forecasting, and early warning detection systems.")

        if not st.session_state.data_in_db:
            st.warning(" No active dataset found. Please upload a CSV on the 'Data Ingestion' workspace to view predictive metrics.")
        else:
            metrics = get_dashboard_metrics()
            df_scores, df_high_risk = get_predictive_data()

            if metrics:
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Database Health Index", f"{metrics['avg_score']}%")
                high_risk_count = len(df_high_risk) if df_high_risk is not None else 0
                rc2.metric("Flagged High-Risk Cases", f"{high_risk_count}", "Match score >= 95")
                rc3.metric("Filing Risk Confidence", "High" if metrics['avg_score'] >= 85 else "Normal")

                st.markdown("<br>", unsafe_allow_html=True)

                # New tabs for enhancedForecasting
                pred_tab1, pred_tab2, pred_tab3 = st.tabs([
                    " Risk Index Distribution",
                    " Proactive Planning (Time Series)",
                    " Early Warning Radars"
                ])

                # Tab 1: Risk Index Distribution
                with pred_tab1:
                    st.markdown("### Match Score Distribution Index")
                    if df_scores is not None and not df_scores.empty:
                        df_scores_chart = df_scores.set_index("Match Score")
                        st.bar_chart(df_scores_chart, width="stretch")
                    else:
                        st.caption("Match score parameters not found in schema.")

                    st.markdown("---")
                    st.markdown("####  Flagged Critical Entities (Score >= 95)")
                    st.markdown("These entities require immediate verification based on database matching scores.")
                    if df_high_risk is not None and not df_high_risk.empty:
                        st.dataframe(df_high_risk, width="stretch", hide_index=True)
                    else:
                        st.success("No records match the high-risk criteria (Match Score >= 95). All cases are within standard matching parameters.")

                # Tab 2: Proactive Planning - Time Series Forecasting
                with pred_tab2:
                    st.markdown("###  Proactive Planning: Time Series Forecasting")
                    st.markdown("Forecast future filing trends by channels and sources for strategic planning.")

                    fcol1, fcol2 = st.columns(2)

                    with fcol1:
                        st.markdown("#### Channel-to-Forecast Trends")
                        st.caption("Historical and projected filing patterns by channel.")
                        df_channel_forecast = get_timeseries_forecast_channels()
                        if df_channel_forecast is not None and not df_channel_forecast.empty:
                            # Group by channel for trend visualization
                            for channel in df_channel_forecast['Channel'].unique():
                                channel_data = df_channel_forecast[df_channel_forecast['Channel'] == channel].set_index('Year')
                                st.write(f"**{channel}**")
                                st.line_chart(channel_data[['Filing Count']], width="stretch")
                        else:
                            st.info("Channel forecast data not available. Check if channel/source columns exist.")

                    with fcol2:
                        st.markdown("#### Source-to-Forecast Market Share")
                        st.caption("Project market share trends for each filing source.")
                        df_source_forecast = get_timeseries_forecast_sources()
                        if df_source_forecast is not None and not df_source_forecast.empty:
                            # Show recent trends
                            recent_years = df_source_forecast['Year'].unique()[:3]
                            for year in sorted(recent_years, reverse=True):
                                year_data = df_source_forecast[df_source_forecast['Year'] == year]
                                st.write(f"**{year} Market Share**")
                                st.bar_chart(year_data.set_index('Source')[['Market Share %']], width="stretch")
                        else:
                            st.info("Source forecast data not available. Check if source/client columns exist.")

                    st.markdown("---")
                    st.markdown("####  Detailed Forecast Data")
                    forecast_col1, forecast_col2 = st.columns(2)
                    with forecast_col1:
                        st.caption("**Channel Forecast Details**")
                        if df_channel_forecast is not None and not df_channel_forecast.empty:
                            st.dataframe(df_channel_forecast, width="stretch", hide_index=True)

                    with forecast_col2:
                        st.caption("**Source Forecast Details**")
                        if df_source_forecast is not None and not df_source_forecast.empty:
                            st.dataframe(df_source_forecast, width="stretch", hide_index=True)

                # Tab 3: Early Warning Radars
                with pred_tab3:
                    st.markdown("###  Early Warning Radar System")
                    st.markdown("Detect anomalies and risk indicators with configurable data range filters.")

                    # Data range filter
                    st.markdown("####  Analysis Filter: Data Range")
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    
                    with filter_col1:
                        use_date_filter = st.checkbox("Enable Date Range Filter", value=False)
                    
                    date_start = None
                    date_end = None
                    if use_date_filter:
                        with filter_col2:
                            date_start = st.date_input("Start Date", value=None)
                        with filter_col3:
                            date_end = st.date_input("End Date", value=None)
                    
                    # Convert dates to string format if provided
                    if date_start:
                        date_start = date_start.strftime('%Y-%m-%d')
                    if date_end:
                        date_end = date_end.strftime('%Y-%m-%d')

                    st.markdown("---")
                    
                    # Get early warning data
                    alerts = get_early_warning_alerts(date_start, date_end)

                    if alerts:
                        # High-Risk Cluster Detection
                        if 'high_risk' in alerts and alerts['high_risk'] is not None and not alerts['high_risk'].empty:
                            st.markdown("####  High-Risk Case Clustering")
                            st.dataframe(alerts['high_risk'], width="stretch", hide_index=True)

                        # Spike Detection (Anomaly Detection)
                        if 'spike_detection' in alerts and alerts['spike_detection'] is not None and not alerts['spike_detection'].empty:
                            st.markdown("####  Filing Volume Spike Detection (Monthly Anomalies)")
                            st.caption("Identify unusual filing patterns and sudden increases in volume.")
                            df_spikes = alerts['spike_detection']
                            if '% Change' in df_spikes.columns:
                                st.dataframe(df_spikes, width="stretch", hide_index=True)
                                # Visualization
                                spike_chart = df_spikes.set_index('Month')[['Monthly Filings']]
                                st.line_chart(spike_chart, width="stretch")

                        # Chapter Distribution Anomalies
                        if 'chapter_anomalies' in alerts and alerts['chapter_anomalies'] is not None and not alerts['chapter_anomalies'].empty:
                            st.markdown("####  Chapter Distribution Anomalies")
                            st.caption("Detect unusual patterns in bankruptcy chapter distributions.")
                            df_anomalies = alerts['chapter_anomalies']
                            st.dataframe(df_anomalies, width="stretch", hide_index=True)
                            # Pie chart visualization
                            st.bar_chart(df_anomalies.set_index('chapter')[['% Distribution']], width="stretch")
                    else:
                        st.info("Early warning data not available. Ensure date columns exist in the dataset.")
            else:
                st.error("Error retrieving predictive analytics from database.")

    # -----------------------------------------------------------------
    # TAB 3:AI Assistant
    # -----------------------------------------------------------------
    elif selected_tab == "AI Assistant":
        st.subheader(" AI Transactional Assistant")
        st.markdown("Ask natural language queries about the filings database. The AI will translate it into a safe SQL query, validate it, execute it, and generate visual insights.")

        # Message history container
        chat_box = st.container(height=500)

        with chat_box:
            for idx, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    content = msg.get("content")

                    # Handle structured dictionary content
                    if isinstance(content, dict):
                        if content.get("type") == "table":
                            st.markdown(content.get("message", "Result:"))
                            df_res = pd.DataFrame(content.get("data", []))
                            st.dataframe(df_res, width="stretch", hide_index=True)
                        elif content.get("type") == "text":
                            st.markdown(content.get("message", ""))
                        else:
                            st.write(content)
                    else:
                        st.markdown(content)

                    # Render SQL details for assistant messages
                    if msg["role"] == "assistant":
                        if "sql_query" in msg:
                            with st.expander("🔍 Generated SQL"):
                                st.code(msg["sql_query"], language="sql")

                        if "validation_result" in msg:
                            vr = msg["validation_result"]
                            with st.expander(" Validation Report"):
                                st.markdown(f"**Status:** `{vr['status']}`")
                                st.markdown(f"**Explanation:** {vr['explanation']}")

                        # Render dataframe if stored inside the chat history
                        if "dataframe" in msg and msg["dataframe"] is not None:
                            st.dataframe(msg["dataframe"], width="stretch", hide_index=True)

                        # Render insights if they are active
                        if msg.get("has_insights") and msg.get("user_query"):
                            with st.expander("📊 View Insights and Charts", expanded=True):
                                generate_insights(
                                    msg["dataframe"],
                                    chart_type=msg.get("chart_type", "auto"),
                                    user_query=msg["user_query"]
                                )

        st.divider()

        # Chat input
        user_input = st.chat_input("Query database (e.g. 'Show total records by state' or 'Plot a pie chart of chapter breakdown')...")

        if user_input:
            if not st.session_state.data_in_db:
                st.error("No active dataset found in database. Please upload a CSV first.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()

        # Process last user message if exists and chatbot has not answered it yet
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            query_text = last_msg["content"]

            with st.chat_message("assistant"):
                with st.spinner("Processing request..."):
                    _handle_user_query(query_text, schema, active_schema)
                st.rerun()


# =============================================================================
# USER QUERY HANDLER
# =============================================================================

def _handle_user_query(user_query, schema, active_schema):
    """Handle user queries by generating, validating, executing SQL and generating insights"""
    try:
        # Use active schema (from database) or default schema
        working_schema = active_schema if active_schema else schema
        
        logger.info("Processing user query: %s", user_query)
        
        # Step 1: Generate SQL from natural language question
        with st.spinner("🔧 Step 1: Generating SQL from your question..."):
            sql_query = generate_sql_from_question(user_query, working_schema, st.session_state.conversation_memory)
        
        if not sql_query:
            error_msg = "❌ Failed to generate SQL query from your question. Please try rephrasing."
            st.error(error_msg)
            _append_conversation_memory(user_query, "", None, answer=error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return
        
        # Display generated SQL
        with st.expander(" Generated SQL", expanded=False):
            st.code(sql_query, language="sql")
        
        # Step 2: Validate SQL with judge LLM
        with st.spinner("🔍 Step 2: Validating SQL with AI judge..."):
            validation_result = validate_sql_with_judge(sql_query, user_query, working_schema)
        
        # Display validation result
        with st.expander(" SQL Validation Report", expanded=False):
            st.markdown(f"**Status:** `{validation_result['status']}`")
            st.markdown(f"**Explanation:** {validation_result['explanation']}")
        
        # Check if validation passed and use appropriate query
        final_query = sql_query
        
        if not validation_result['is_valid']:
            # Auto-repair was attempted
            if validation_result.get('repaired_query'):
                st.warning(f" SQL validation indicated issues: {validation_result['explanation']}")
                st.info(" Using auto-repaired query...")
                final_query = validation_result['repaired_query']
            else:
                error_msg = f" SQL validation failed: {validation_result['explanation']}"
                st.error(error_msg)
                _append_conversation_memory(user_query, sql_query, None, validation_result, answer=error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sql_query": sql_query,
                    "validation_result": validation_result
                })
                return
        else:
            st.success("✓ SQL validation passed!")
        
        # Step 3: Execute the SQL query
        with st.spinner("⚙️ Step 3: Executing query..."):
            result_df = execute_sql_query(final_query)
        
        if result_df is None:
            error_msg = " Error executing the SQL query."
            st.error(error_msg)
            _append_conversation_memory(user_query, final_query, None, validation_result, answer=error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sql_query": final_query,
                "validation_result": validation_result
            })
            return
        
        if len(result_df) == 0:
            msg = "✓ Query executed successfully, but no records matched your criteria."
            st.info(msg)
            _append_conversation_memory(user_query, final_query, [], validation_result, answer=msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df
            })
            return
        
        # Display results
        success_msg = f" Query executed successfully! Found {len(result_df)} records."
        st.success(success_msg)
        st.dataframe(result_df, width='stretch', hide_index=True)
        
        # Step 4: Generate insights if applicable
        should_insights = should_generate_insights(user_query, result_df)
        chart_type = detect_chart_type(user_query)
        
        if should_insights and len(result_df) >= 2:
            with st.expander(" View Insights and Charts", expanded=True):
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        # Store in conversation memory
        _append_conversation_memory(user_query, final_query, result_df.to_dict('records'), validation_result)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type
        })
        
        logger.info("Query processed successfully | rows=%d | chart_type=%s", len(result_df), chart_type)
        
    except Exception as e:
        error_msg = f" Unexpected error: {str(e)}"
        logger.exception("Error in user query handling: %s", e)
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    try:
        logger.info("Starting Bankruptcy application")
        main()
    except Exception as e:
        logger.exception("Fatal error in main application: %s", e)
        raise