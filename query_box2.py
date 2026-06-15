import sqlite3
import logging
import sys
import json
import re
import hashlib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from config import call_llm, call_llm_with_cache, count_token_usage
from dotenv import load_dotenv
from insights_generator import generate_insights, generate_summary

# =============================================================================
# CONFIG & INITIALIZATION
# =============================================================================

# Suggested Questions for Quick Access
# You can modify these questions to customize the suggestions shown to users
SUGGESTED_QUESTIONS = [
    "Give me insights about top three organisations with critical severity backstops?",
    "How many total high severity backstops in Engineering organisation?",
    "Give me a insights on major vulnerability titles contributing to defects?",
    "Which organisation has the most high backstop issues?",
]

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vulnerability_genbi.log", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("vulnerability_genbi")

hide_toolbar_css = """
<style>
    /* Hides the right-side toolbar (Deploy + 3-dots) */
    [data-testid="stToolbar"] {
        display: none;
    }
</style>
"""
st.markdown(hide_toolbar_css, unsafe_allow_html=True)

st.set_page_config(
    page_title="Vulnerability GenBI",
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


def display_suggested_questions(questions, table = "uploaded_data"):
    """Display dynamic question suggestions as clickable buttons"""
    # Initialize session state for suggested question clicks
    try:
        if "suggested_question_selected" not in st.session_state:
            st.session_state.suggested_question_selected = None
        # Display questions in a responsive grid layout
        cols = st.columns(min(2, len(questions)))

        for idx, question in enumerate(questions):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                if st.button(
                    question,
                    key=f"suggested_q_{table + str(idx)}",
                    use_container_width=True,
                    help="Click to ask this question"
                ):
                    st.session_state.suggested_question_selected = question
                    st.rerun()
    except Exception as e:
        logger.error("Error initializing session state: %s", e)


def get_db_table_name():
    """Detect active table name in SQLite database"""
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='uploaded_data' OR name='vulnerability_data')")
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


def _clean_sql_query(sql: str) -> str:
    """Clean extracted SQL query of trailing/embedded unwanted characters"""
    if not sql:
        return ""
    
    sql = sql.strip()
    
    # Remove trailing semicolons, braces, and other SQL-invalid chars
    sql = re.sub(r'[;}\s]+$', '', sql)
    
    # Remove SQL line comments (-- ...)
    sql = re.sub(r'--.*?(?=\n|$)', '', sql)
    
    # Remove SQL block comments (/* ... */)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.S)
    
    # Normalize whitespace (multiple spaces to single space)
    sql = re.sub(r'\s+', ' ', sql)
    
    sql = sql.strip()
    
    # Final cleanup: remove any trailing braces or semicolons that might have been left
    sql = re.sub(r'[;}\s]+$', '', sql)
    
    return sql


def extract_sql_from_response(text: str) -> str:
    """Extract SQL statement from LLM response with robust edge case handling"""
    try:
        text = text.strip()
        
        # Strategy 1: Look for JSON structure with proper handling
        # Use non-greedy matching to find JSON objects
        # Try to find all potential JSON objects and parse them
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for m in re.finditer(json_pattern, text, re.S):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    # Try 'sql' key first
                    if 'sql' in obj and obj['sql']:
                        sql = _clean_sql_query(obj['sql'])
                        if sql and sql.upper().startswith('SELECT'):
                            return sql
                    # Try 'CORRECTED_QUERY' key
                    elif 'CORRECTED_QUERY' in obj and obj['CORRECTED_QUERY']:
                        sql = _clean_sql_query(obj['CORRECTED_QUERY'])
                        if sql and sql.upper().startswith('SELECT'):
                            return sql
            except (json.JSONDecodeError, ValueError):
                # Not a valid JSON object, try next pattern
                pass
        
        # Strategy 2: Look for SQL code block formats (various markdown styles)
        sql_block_patterns = [
            r'```sql\n(.*?)```',           # Standard with newline
            r'```\s*sql\s*\n(.*?)```',    # SQL with optional spaces
            r'```\s*sql\s*(.*?)```',      # SQL without required newline after marker
            r'`{3,}\s*sql\s*\n(.*?)`{3,}',  # Multiple backticks
        ]
        
        for pattern in sql_block_patterns:
            m = re.search(pattern, text, re.S | re.I)
            if m:
                sql = _clean_sql_query(m.group(1))
                if sql and sql.upper().startswith('SELECT'):
                    return sql
        
        # Strategy 3: Extract SELECT statement (most lenient approach)
        # Match from SELECT keyword to a natural boundary
        # Boundary can be: semicolon, closing brace, double newline, or end of string
        m = re.search(r'(SELECT\s+[\s\S]*?)(?:;|\}|\n\n|$)', text, re.I)
        if m:
            sql = _clean_sql_query(m.group(1))
            if sql and sql.upper().startswith('SELECT'):
                return sql
        
        return ""
    except Exception as e:
        logger.exception("Error extracting SQL: %s", e)
        return ""


def generate_sql_from_question(user_question, schema, conversation_memory=None, token_usage=None):
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
            "STRICTLY For date related queries filtering , you should only use the strftime function and date type column names mentioned in the schema. Do not use any other functions or column names that are not in the schema.\n"
            "STRICTLY use ONLY the column names explicitly mentioned in the Database schema below. Do not use, guess, or hallucinate any other column names.\n"
            "Return only valid JSON with a single key `sql` whose value is the SQL string.\n"
            "Do NOT include explanations or additional fields. Ensure the SQL is compatible with SQLite.\n\n"
            "Database schema (JSON):\n" + json.dumps(schema['columns']) + "\n\n"
            "Table name: " + schema['table_name'] + "\n\n"
        )
        # if memory_context:
        #     prompt += f"\nConversation context:\n{memory_context}\n"

        prompt += (
            "User request: \"" + user_question + "\"\n\n"
            "If the request cannot be represented as a single SQL SELECT query and do not include ';' at the end of the query, return an empty string for `sql`."
        )
        response = call_llm_with_cache(prompt)
        sql_query = extract_sql_from_response(response)
        if token_usage is not None:
            token_usage.update(count_token_usage(prompt, response))
        logger.info("SQL generated | length=%d | query=%s", len(sql_query), sql_query[:100] if sql_query else "EMPTY")
        return sql_query
    except Exception as e:
        logger.exception("Error generating SQL from question: %s", e)
        return ""


def validate_sql_with_judge(sql_query, user_question, schema, token_usage=None):
    """Use an LLM as a judge to validate if the generated SQL is correct"""
    sql_upper = sql_query.strip().upper()

    # Block writing operations
    ddl_dml = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'RENAME', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'REPLACE']
    for keyword in ddl_dml:
        if keyword in sql_upper:
            return {
                "is_valid": False,
                "status": "BLOCKED",
                "explanation": f"Write operation keyword '{keyword}' is forbidden. Only SELECT operations are allowed.",
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
            f"- {col['Column Name']} ({col['Data Type']}): {col['Description']}"
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

        response = call_llm_with_cache(prompt)
        if token_usage is not None:
            token_usage.update(count_token_usage(prompt, response))
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
# TEMPORARY TABLE MANAGEMENT FOR FOLLOW-UP QUESTIONS
# =============================================================================

def create_temporary_table_from_dataframe(result_df, source_query):
    """Create a temporary SQLite table from a result dataframe for follow-up queries.
    
    Returns a tuple of (table_name, schema_dict) for use in follow-up queries.
    """
    try:
        import time
        table_name = f"temp_result_{int(time.time() * 1000) % 1000000}"
        
        # Connect and create temporary table
        conn = sqlite3.connect('data.db')
        result_df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        
        # Build schema for this temporary table
        temp_schema = {
            "table_name": table_name,
            "columns": [
                {
                    "Column Name": col,
                    "Data Type": str(result_df[col].dtype),
                    "Description": f"Column from previous query result"
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


def is_followup_question(user_query, conversation_history):
    """Detect if the user is asking a follow-up question.
    
    Returns True if:
    - There's an active temporary table (previous result set exists)
    - The query looks like it's asking about the previous results (e.g., contains "that", "those", "records", etc.)
    """
    if not st.session_state.temp_table_name:
        return False
    
    followup_keywords = ['that', 'those', 'these', 'records', 'results', 'data', 'rows', 'items', 'from that', 'from there', 'among those', 'above', 'convert']
    query_lower = user_query.lower()
    return any(kw in query_lower for kw in followup_keywords)

def _generate_follow_up_questions(schema, memory):
    """Generate follow-up questions based on conversation memory"""
    follow_up_questions = []
    
    try:
        # Format memory as a readable string
        memory_str = json.dumps(memory, indent=2) if isinstance(memory, dict) else str(memory)
        
        # Build the prompt
        prompt = (
            "Based on the following conversation history between a user and an assistant about a vulnerability dataset, generate at least 4 relevant follow-up questions that the user might ask next. Focus on questions that can be convertable to sqlite compatible queries. The generated questions should be queried on database with below schema.\n\n"
            "Database schema (JSON):"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Conversation History:\n{memory_str}\n\n"
            "Generate simple questions only based on the conversation history. Use the schema only to ensure question are relevant to the data. And remember these question should be converted to SQL queries later, so avoid questions that are too complex or require multi-step reasoning.\n\n"
            "If you are not able to generate 4 questions based on the conversation history, generate as many as you can up to 4. Do not generate questions that are not relevant to the conversation history. Do not include any explanations or additional text, only return a JSON array of question strings.\n\n"
            "Return ONLY a JSON array of exactly 4 question strings, like: [\"question1\", \"question2\", \"question3\", \"question4\"]"
        )
        
        # Call LLM to generate questions
        response = call_llm(prompt)
        logger.info("Follow-up questions response received | length=%d", len(response))
        
        # Try to extract JSON array from response
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group(0))
                if isinstance(questions_data, list):
                    follow_up_questions = [str(q).strip() for q in questions_data if q and isinstance(q, str)]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from follow-up questions response")
        
        # Fallback: extract numbered questions (e.g., "1. Question here?")
        if not follow_up_questions:
            lines = response.split('\n')
            for line in lines:
                match = re.match(r'^\d+\.\s+(.+)$', line.strip())
                if match:
                    question = match.group(1).strip()
                    if question:
                        follow_up_questions.append(question)
        
        # Ensure we have at most 4 questions
        follow_up_questions = follow_up_questions[:4]
        
        logger.info("Extracted %d follow-up questions from LLM response", len(follow_up_questions))
        return follow_up_questions
        
    except Exception as e:
        logger.exception("Error generating follow-up questions: %s", e)
        return []

def _initialize_conversation_memory():
    """Initialize conversation memory structure"""
    return {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
    }


def _append_conversation_memory(user_query, sql_query, records, summary=None, answer=None):
    """Append structured conversation memory"""
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory
        memory_entry = {
                "user_question": user_query,
                "sql_query": sql_query,
                "records": records,
                "record_count": len(records) if records else 0,
                "summary": summary,
            }

        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query

        if records:
            memory["last_assistant_response"] = f"SQL: {sql_query} | Records: {records if records else 0}"
        else:
            memory["last_assistant_response"] = f"SQL: {sql_query} | No records returned or query failed."

        # Keep only last 8 conversations
        if len(memory["history"]) > 8:
            memory["history"] = memory["history"][-8:]

    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)

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
                "content": " **Welcome to Vulnerability GenBI Assistant!** Ask me any natural language questions about the loaded Vulnerability data."
            }
        ]

    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}

    # Temporary table management for follow-up questions
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

    # =========================================================================
    # SIDEBAR NAVIGATION (Light Gray Layout Theme)
    # =========================================================================
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1rem 0; text-align: center;">
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                     Vulnerability GenBI Assistant 
                </span>
            </div>
            <hr style="margin: 0.5rem 0 1.5rem 0 !important; border-top: 1px solid #cbd5e1;">
            """,
            unsafe_allow_html=True
        )
        selected_tab = st.radio(
            "Select Workspace",
            ["Data", "Query Box"],
            label_visibility="collapsed"
        )

    # Header section
    #st.markdown('<div class="main-title"> Vulnerability GenBI Assistant</div>', unsafe_allow_html=True)

    # =========================================================================
    # DATA WORKSPACE TABS SWITCH
    # =========================================================================

    # -----------------------------------------------------------------
    # TAB 0:Data (Separated Panel)
    # -----------------------------------------------------------------
    if selected_tab == "Data":
        st.subheader("Data & Repository Management")
        st.caption("Upload transactional files dynamically to populate the SQLite analytics structure.")

        uploaded_file = st.file_uploader("Upload CSV database...", type=["csv"], help="Upload backstops CSV data. Columns will be parsed dynamically.")

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
                        <span style="color: #15803d; font-weight: 500; font-size: 1rem;">Active Database</span>
                        <div style="color: #166534; font-size: 0.9rem; font-weight: 500; margin-top: 0.4rem;">
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
    elif selected_tab == "Query Box":
        st.subheader("AI Assistant")
        # Message history container
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
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
                        st.dataframe(df_res, width="content", hide_index=True)
                    elif content.get("type") == "text":
                        st.markdown(content.get("message", ""))
                    else:
                        st.write(content)
                else:
                    st.markdown(content)

                # Render SQL details for assistant messages
                if msg["role"] == "assistant":
                    if "validation_result" in msg:
                        vr = msg["validation_result"]

                    # Render dataframe if stored inside the chat history
                    if "dataframe" in msg and msg["dataframe"] is not None:
                        st.dataframe(msg["dataframe"], width="content", hide_index=True)
                    
                    if "summary" in msg and msg["summary"] is not None:
                        st.write(msg['summary'])

                    # Render insights if they are active
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
                            if st.button("ℹ️", key=token_info_button_key, help="Click to show token usage details"):
                                st.session_state[token_info_state_key] = not st.session_state[token_info_state_key]
                                show_token_usage = st.session_state[token_info_state_key]

                    if show_token_usage and "token_usage" in msg and msg["token_usage"]:
                        with st.expander("ℹ️ Token usage", expanded=True):
                            for step_name, usage in msg["token_usage"].items():
                                if isinstance(usage, dict) and usage:
                                    st.markdown(
                                        f"**{step_name.title()}**: Input `{usage['input_tokens']}` | Output `{usage['output_tokens']}` | Total `{usage['total_tokens']}`"
                                    )

        if st.session_state.temp_table_name:
            suggested_questions = _generate_follow_up_questions(schema, st.session_state.conversation_memory)
            display_suggested_questions(suggested_questions, st.session_state.temp_table_name)
        else:
            display_suggested_questions(SUGGESTED_QUESTIONS)
        
        # Chat input
        user_input = st.chat_input("Ask a question about the vulnerability data...")
        
        # Check if a suggestion was clicked
        if st.session_state.get("suggested_question_selected"):
            user_input = st.session_state.suggested_question_selected
            st.session_state.suggested_question_selected = None

        if user_input:
            if not st.session_state.data_in_db:
                st.error("No active dataset found in database. Please upload a CSV first.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})

        # Process last user message if exists and chatbot has not answered it yet
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            query_text = last_msg["content"]

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    _handle_user_query(query_text, schema)


# =============================================================================
# USER QUERY HANDLER
# =============================================================================

def _handle_user_query(user_query, schema):
    """Handle user queries by generating, validating, executing SQL and generating insights"""
    try:
        logger.info("Processing user query: %s", user_query)
        
        # Check if this is a follow-up question (asking about previous results)
        is_followup = is_followup_question(user_query, st.session_state.conversation_memory)
        
        # Determine which schema to use
        if is_followup and st.session_state.temp_table_schema:
            working_schema = st.session_state.temp_table_schema
            #st.info("**Querying previous result set**")
            logger.info("Using temporary table for follow-up query | table=%s", working_schema.get('table_name'))
        else:
            # First query or regular query
            working_schema =  schema
            
            # If user explicitly asks to clear or start fresh, drop temporary table
            if any(kw in user_query.lower() for kw in ['clear', 'reset', 'new query', 'fresh', 'different']):
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
        if user_query in st.session_state.query_cache:
            cached_response = st.session_state.query_cache[user_query]
            st.write("Fetched results from cache.")
            st.dataframe(cached_response["dataframe"], width='content', hide_index=True)
            if cached_response.get("summary", None):
                st.write(cached_response["summary"])
            if cached_response.get("has_insights"):
                with st.expander("View Insights and Charts", expanded=True):
                    generate_insights(
                        cached_response["dataframe"],
                        chart_type=cached_response.get("chart_type", "auto"),
                        user_query=user_query,
                    )
            _append_conversation_memory(
                user_query,
                cached_response["sql_query"],
                cached_response["records"],
                cached_response.get("summary", None),
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
                "summary": cached_response.get("summary", None),
            })
            return

        # Step 1: Generate SQL from natural language question
        sql_query = generate_sql_from_question(
                user_query,
                working_schema,
                st.session_state.conversation_memory,
                token_usage=generation_token_usage,
            )

        if not sql_query:
            error_msg = "Failed to generate SQL query from your question. Please try rephrasing."
            st.error(error_msg)
            _append_conversation_memory(user_query, "", [], None, answer=error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return
        
        # Display generated SQL
        # with st.expander(" Generated SQL", expanded=False):
        #     st.code(sql_query, language="sql")
        
        # Step 2: Validate SQL with judge LLM
        validation_result = validate_sql_with_judge(sql_query, user_query, working_schema)
        
        # Display validation result
        # with st.expander(" SQL Validation Report", expanded=False):
        #     st.markdown(f"**Status:** `{validation_result['status']}`")
        #     st.markdown(f"**Explanation:** {validation_result['explanation']}")
        
        # Check if validation passed and use appropriate query
        final_query = sql_query
        
        if not validation_result['is_valid']:
            # Auto-repair was attempted
            if validation_result.get('repaired_query'):
                st.warning(f" SQL validation indicated issues: {validation_result['explanation']}")
                st.info(" Using auto-repaired query...")
                final_query = validation_result['repaired_query']
            else:
                st.error(validation_result['explanation'])
                #_append_conversation_memory(user_query, sql_query, None, validation_result, answer=validation_result['explanation'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": validation_result['explanation'],
                    "sql_query": sql_query,
                    "validation_result": validation_result
                })
                return

        
        # Step 3: Execute the SQL query
        result_df = execute_sql_query(final_query)
        
        if result_df is None:
            error_msg = " Error executing the SQL query."
            st.error(error_msg)
            _append_conversation_memory(user_query, final_query, [], None, answer = error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sql_query": final_query,
                "validation_result": validation_result
            })
            return
        
        if len(result_df) == 0:
            msg = "Query executed successfully, but no records matched your criteria."
            st.info(msg)
            _append_conversation_memory(user_query, final_query, [], None, answer=msg)
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
                "summary": None,
            })
            st.session_state.query_cache[user_query] = {
                "message": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "records": [],
                "user_query": user_query,
                "has_insights": False,
                "chart_type": "auto",
                "token_usage": token_usage,
                "summary": None,
            }
            return
        
        # Display results
        success_msg = f" Fetched records successfully. "
        st.write(success_msg)
        st.dataframe(result_df, width='content', hide_index=True)
        summary = generate_summary(result_df, user_query)
        if summary:
            st.write(summary)
        # For non-follow-up queries, create a temporary table for potential follow-up questions
        if not is_followup:
            temp_table_name, temp_schema = create_temporary_table_from_dataframe(result_df, final_query)
            if temp_table_name and temp_schema:
                st.session_state.temp_table_name = temp_table_name
                st.session_state.temp_table_schema = temp_schema
                st.session_state.temp_table_source_query = final_query
                st.session_state.temp_table_dataframe = result_df
        
        # Step 4: Generate insights if applicable
        should_insights = should_generate_insights(user_query, result_df)
        chart_type = detect_chart_type(user_query)
        
        if should_insights and len(result_df) >= 2:
            with st.expander("View Insights and Charts", expanded=True):
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        # Store in conversation memory
        _append_conversation_memory(user_query, final_query, result_df.to_dict('records'), summary, answer=success_msg)
        token_info_state_key = f"token_info_state"
        token_info_button_key = f"token_info_btn"
        if token_info_state_key not in st.session_state:
            st.session_state[token_info_state_key] = False
        show_token_usage = st.session_state[token_info_state_key]
        cols = st.columns([0.92, 0.08])
        with cols[1]:
            if st.button("ℹ️", key=token_info_button_key, help="Click to show token usage details"):
                st.session_state[token_info_state_key] = not st.session_state[token_info_state_key]
                show_token_usage = st.session_state[token_info_state_key]

        if show_token_usage and "token_usage" in msg and msg["token_usage"]:
            with st.expander("ℹ️ Token usage", expanded=True):
                for step_name, usage in msg["token_usage"].items():
                    if isinstance(usage, dict) and usage:
                        st.markdown(
                        f"**{step_name.title()}**: Input `{usage['input_tokens']}` | Output `{usage['output_tokens']}` | Total `{usage['total_tokens']}`")
        
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
            "summary": summary,
        })
        st.session_state.query_cache[user_query] = {
            "message": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "records": result_df.to_dict('records'),
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
            "summary": summary,
        }
        # if st.session_state.temp_table_name and st.session_state.temp_table_dataframe is not None:
        #     temp_df = st.session_state.temp_table_dataframe
        #     cols = st.columns([0.7, 0.3])
        #     with cols[0]:
        #         st.markdown(
        #             f"**Active Result Set**: {len(temp_df)} records from previous query | "
        #             f"{len(temp_df.columns)} columns"
        #         )
        #     with cols[1]:
        #         if st.button("Clear & Reset", key="clear_temp_table", use_container_width=True):
        #             drop_temporary_table(st.session_state.temp_table_name)
        #             st.session_state.temp_table_name = None
        #             st.session_state.temp_table_schema = None
        #             st.session_state.temp_table_source_query = None
        #             st.session_state.temp_table_dataframe = None
        #             st.success("Cleared temporary table. Ready for a new query.")
                # Display suggested questions
        #display_suggested_questions(st.session_state.temp_table_name)       
        logger.info("Query processed successfully | rows=%d | chart_type=%s | temp_table=%s", 
                    len(result_df), chart_type, st.session_state.temp_table_name or "none")
        
    except Exception as e:
        error_msg = f" Unexpected error: {str(e)}"
        logger.exception("Error in user query handling: %s", e)
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    st.rerun()


if __name__ == "__main__":
    try:
        logger.info("Starting Vulnerability application")
        main()
    except Exception as e:
        logger.exception("Fatal error in main application: %s", e)
        raise