import sqlite3
import logging
import sys
import json
import re
import pandas as pd
import streamlit as st
from config import  call_llm_with_tokens, call_llm_haiku_with_tokens
from dotenv import load_dotenv
from insights_generator import generate_insights

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
       logging.FileHandler("./bankruptcy3_genbi.log", mode="a", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("bankruptcy3_genbi")

st.set_page_config(
    page_title="Bankruptcy GenBI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.markdown(
#     """
#     <style>
#     /* Root theme variables */
#     :root {
#         --primary-bg: #fdfdfd;
#         --primary-color: #6366f1;
#         --primary-dark: #4f46e5;
#         --accent-color: #f5f3ff;
#         --success-color: #22c55e;
#         --error-color: #f87171;
#         --shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
#         --shadow-lg: 0 10px 25px rgba(14, 165, 233, 0.2);
#         --border-radius: 12px;
#     }

#     /* Main container styling */
#     .main {
#         background-color: #edfcfc;
#         background-image: 
#             radial-gradient(circle at 20% 50%, rgba(15, 229, 255, 0.08) 0%, transparent 50%),
#             radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%);
#     }

#     .main .block-container {
#         padding-top: 2.5rem;
#         padding-bottom: 2.5rem;
#         padding-left: 3rem;
#         padding-right: 3rem;
#         max-width: 1400px;
#     }

#     /* Sidebar styling */
#     .css-1d391kg {
#         background: #f7ffff;
#         border-right: 2px solid #d1faf8;
#     }

#     /* Header styling */
#     h1, h2, h3 {
#         color: #0284c7;
#         font-weight: 700;
#         letter-spacing: -0.5px;
#     }

#     h1 {
#         font-size: 2.5rem;
#         margin-bottom: 0.5rem;
#         background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }

#     h2 {
#         font-size: 1.75rem;
#         margin-top: 1.5rem;
#         margin-bottom: 1rem;
#     }

#     h3 {
#         font-size: 1.25rem;
#         margin-top: 1.25rem;
#         margin-bottom: 0.75rem;
#     }

#     /* Success message styling */
#     .stSuccess {
#         background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
#         border: 2px solid #10b981;
#         border-radius: 12px;
#         padding: 1.25rem;
#         color: #065f46;
#         box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
#         font-weight: 500;
#     }

#     /* Error message styling */
#     .stError {
#         background: linear-gradient(135deg, #fef2f2 0%, #fef2f2 100%);
#         border: 2px solid #ef4444;
#         border-radius: 12px;
#         padding: 1.25rem;
#         color: #7f1d1d;
#         box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
#         font-weight: 500;
#     }

#     /* Warning message styling */
#     .stWarning {
#         background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
#         border: 2px solid #f59e0b;
#         border-radius: 12px;
#         padding: 1.25rem;
#         color: #78350f;
#         box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
#         font-weight: 500;
#     }

#     /* Info message styling */
#     .stInfo {
#         background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
#         border: 2px solid #0ea5e9;
#         border-radius: 12px;
#         padding: 1.25rem;
#         color: #0c4a6e;
#         box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
#         font-weight: 500;
#     }

#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
#         color: white;
#         border: none;
#         border-radius: 10px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         font-size: 0.95rem;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
#         letter-spacing: 0.3px;
#     }

#     .stButton > button:hover {
#         background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
#         box-shadow: 0 8px 20px rgba(14, 165, 233, 0.4);
#         transform: translateY(-2px);
#     }

#     .stButton > button:active {
#         transform: translateY(0);
#     }

#     /* Input styling */
#     .stTextInput > div > div > input,
#     .stTextArea > div > div > textarea,
#     .stSelectbox > div > div > select,
#     .stNumberInput > div > div > input,
#     .stDateInput > div > div > input,
#     .stTimeInput > div > div > input {
#         background-color: #ffffff;
#         border: 2px solid #d1f2f7;
#         border-radius: 10px;
#         color: #0c4a6e;
#         padding: 0.75rem;
#         font-size: 0.95rem;
#         transition: all 0.3s ease;
#     }

#     .stTextInput > div > div > input:focus,
#     .stTextArea > div > div > textarea:focus,
#     .stSelectbox > div > div > select:focus,
#     .stNumberInput > div > div > input:focus,
#     .stDateInput > div > div > input:focus,
#     .stTimeInput > div > div > input:focus {
#         border-color: #0ea5e9;
#         box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
#     }

#     /* Dataframe styling */
#     .stDataframe {
#         border: 1px solid #d1f2f7;
#         border-radius: 12px;
#         overflow: hidden;
#         box-shadow: 0 4px 16px rgba(14, 165, 233, 0.12);
#     }

#     .stDataframe tbody {
#         border-color: #e0f2fe;
#     }

#     .stDataframe thead {
#         background: linear-gradient(90deg, #cffafe 0%, #d1faf8 100%);
#     }

#     .stDataframe tbody tr:hover {
#         background-color: #f0fdfd;
#     }

#     /* Toggle styling */
#     .stCheckbox > label {
#         color: #0c4a6e;
#         font-weight: 500;
#     }

#     /* Expander styling */
#     .streamlit-expanderHeader {
#         background: linear-gradient(90deg, #ecf1f7 0%, #f0f9ff 100%);
#         border: 1px solid #d1f2f7;
#         border-radius: 10px;
#         color: #0284c7;
#         font-weight: 600;
#     }

#     .streamlit-expanderHeader:hover {
#         background: linear-gradient(90deg, #dce9f3 0%, #e0f2fe 100%);
#         box-shadow: 0 2px 8px rgba(14, 165, 233, 0.15);
#     }

#     /* Chat message styling */
#     .chat-message {
#         border-radius: 12px;
#         padding: 1.25rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
#     }

#     .chat-message.user {
#         background: linear-gradient(135deg, #e0f2fe 0%, #ecf1f7 100%);
#         border-left: 4px solid #0284c7;
#     }

#     .chat-message.assistant {
#         background: linear-gradient(135deg, #f0fdfd 0%, #f8fefb 100%);
#         border-left: 4px solid #06b6d4;
#     }

#     /* Sidebar elements */
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #ffffff 0%, #f8fefb 100%);
#     }

#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #ffffff 0%, #f8fefb 100%);
#     }

#     [data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 2rem;
#     }

#     /* File uploader styling */
#     .uploadedFile {
#         background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
#         border: 2px dashed #06b6d4;
#         border-radius: 12px;
#         padding: 1.5rem;
#     }

#     /* Caption and text styling */
#     .stCaption {
#         color: #0c4a6e;
#         font-size: 0.85rem;
#     }

#     p {
#         color: #1e293b;
#         line-height: 1.6;
#         font-size: 0.95rem;
#     }

#     /* Spinner styling */
#     .stSpinner > div {
#         border-color: #0ea5e9;
#     }

#     .stSpinner > div > div {
#         background-color: #0ea5e9;
#     }

#     /* Code block styling */
#     pre {
#         background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
#         border: 1px solid #d1f2f7;
#         border-radius: 10px;
#         padding: 1.25rem;
#         color: #0c4a6e;
#         box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
#     }

#     code {
#         background-color: #f0fdfd;
#         border-radius: 6px;
#         padding: 0.25rem 0.5rem;
#         color: #0284c7;
#         font-size: 0.9rem;
#     }

#     /* Link styling */
#     a {
#         color: #0ea5e9;
#         text-decoration: none;
#         font-weight: 500;
#         transition: all 0.2s ease;
#     }

#     a:hover {
#         color: #0284c7;
#         text-decoration: underline;
#     }

#     /* Tab styling */
#     .stTabs [data-baseweb="tab-list"] button {
#         background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
#         border: 2px solid #d1f2f7;
#         border-radius: 10px 10px 0 0;
#         color: #0c4a6e;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }

#     .stTabs [data-baseweb="tab-list"] button:hover {
#         background: linear-gradient(135deg, #e0f2fe 0%, #dce9f3 100%);
#         border-color: #0ea5e9;
#     }

#     .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
#         background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
#         color: white;
#         border-color: #0284c7;
#     }

#     /* Metric styling */
#     .stMetric {
#         background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
#         border: 2px solid #d1fae5;
#         border-radius: 12px;
#         padding: 1.25rem;
#         box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
#     }

#     /* Column styling */
#     .stColumn {
#         padding: 0 0.75rem;
#     }

#     /* Horizontal line */
#     hr {
#         border-color: #d1f2f7;
#         margin: 1.5rem 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
    <style>
    :root {
        --primary-bg: #edf6f7;
        --card-bg: #ffffff;
        --accent-blue: #0ea5e9;
        --text-main: #1e293b;
        --border-color: #d1e2e4;
    }

    /* Main Container */
    .main {
        background-color: var(--primary-bg);
    }

    .main .block-container {
        padding: 2.5rem 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafb;
        border-right: 1px solid var(--border-color);
    }

    /* Headers */
    h1, h2, h3 {
        color: #0c4a6e;
        font-weight: 700;
    }

    h1 {
        background: linear-gradient(135deg, #0284c7 0%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Input & Select boxes */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select {
        background-color: white !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0ea5e9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        background-color: #0284c7;
        color: white;
        opacity: 0.9;
    }

    /* Dataframe & Tables */
    .stDataframe {
        background-color: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    /* Chat Messages */
    .chat-message.user {
        background-color: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .chat-message.assistant {
        background-color: #e2eff1;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #edfcfc 0%, #f0fdfd 100%);
        padding: 1.22rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
        border: 2px solid #cffafe;
    ">
        <h1 style="
            margin: 0;
            background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.25 rem;
            font-weight: 400;
            letter-spacing: -1px;
        "> Bankruptcy GenBI Assistant</h1>
        <p style="
            margin: 0.75rem 0 0 0;
            color: #0c4a6e;
            font-size: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.2px;
        ">Intelligent Data Analysis & Query System</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_schema():
    """Load database schema from schema.json"""
    try:
        with open('schema.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None


def get_actual_database_schema():
    """Get the actual column structure from the bankruptcy4 table in database"""
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(bankruptcy4);")
        columns_info = cursor.fetchall()
        conn.close()
        
        if not columns_info:
            logger.warning("bankruptcy4 table not found in database")
            return None
        
        # Build schema from actual database columns
        actual_columns = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2] or "text"
            
            # Try to match with schema.json for description, fallback to generic
            description = f"Column {col_name}"
            
            actual_columns.append({
                "name": col_name,
                "type": col_type,
                "description": description
            })
        
        logger.info("Detected %d columns from bankruptcy4 table", len(actual_columns))
        return {
            "schema_version": "1.0",
            "table_name": "bankruptcy4",
            "description": "Dynamic schema from uploaded CSV data",
            "columns": actual_columns
        }
    except Exception as e:
        logger.exception("Error getting actual database schema: %s", e)
        return None
    
def extract_sql_from_response(text: str) -> str:
    """Extract SQL query from LLM response in various formats"""
    try:
        text = text.strip()
        # First, try to find a JSON object
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and 'sql' in obj:
                    logger.debug("SQL extracted from JSON object (sql key)")
                    return obj['sql'].strip()
                else:
                    if isinstance(obj, dict) and 'CORRECTED_QUERY' in obj:
                        logger.debug("SQL extracted from JSON object (CORRECTED_QUERY key)")
                        return obj['CORRECTED_QUERY'].strip()
            except Exception as e:
                logger.debug("Failed to parse JSON: %s", e)
                pass

        # If no JSON, try to find SQL fenced block
        m = re.search(r"```sql\n(.*?)```", text, re.S | re.I)
        if m:
            logger.debug("SQL extracted from code fence block")
            return m.group(1).strip()

        # Fallback: attempt to extract first SELECT ... statement
        m = re.search(r"(SELECT[\s\S]*?);?\s*$", text, re.I)
        if m:
            logger.debug("SQL extracted from SELECT statement")
            return m.group(1).strip()

        logger.warning("No SQL could be extracted from response")
        return ""
    except Exception as e:
        logger.exception("Error extracting SQL from response: %s", e)
        return ""

def follow_up_question(user_question, conversation_memory=None):
    """Generate a follow-up question response using conversation context.
    Returns: {
        'response': str,
        'input_tokens': int,
        'output_tokens': int,
        'total_tokens': int
    }
    """
    try:
        logger.info("Processing follow-up question: %s", user_question)
        memory_context = conversation_memory or _build_memory_context()
        prompt = (
            "You are a precise data analysis assistant.\n"
            "Your task is to answer the user's follow-up question using ONLY the provided CONVERSATION HISTORY.\n\n"
            "Instructions for filtering requests (e.g., 'greater than X', 'less than Y', 'equals Z'):\n"
            "1. Identify the exact field/column to filter on.\n"
            "2. Identify the mathematical or logical condition.\n"
            "3. Evaluate every record in the records against this condition.\n"
            "4. Count the exact number of matching records.\n"
            "5. If you are analysing any filed/column, the records to be considered should be equal to record_count\n"
            "6. Provide a clear summary of the results, including a list of the matched records if appropriate.\n\n"
            "Constraints:\n"
            "- Be 100 percent accurate with comparisons. Do not skip any matching records.\n"
            "- Do not include SQL queries, raw JSON, or code.\n"
            "- Avoid conversational filler like 'Based on the previous query'.\n"
            "- Do not provide any explanations or reasoning steps. Only provide the final answer only based on the conversation history .\n"
            "- If the question is unclear or no data matches, state that simply.\n\n"
            "CONVERSATION HISTORY:\n{history}\n\n"
            "User's follow-up question:\n{user_question}\n\n"
        ).format(
            history=json.dumps(memory_context.get('history', []), indent=2),
            user_question=user_question
        )
        result = call_llm_haiku_with_tokens(prompt)
        logger.info("Follow-up question processed successfully | tokens=%d", result.get('total_tokens', 0))
        return {
            'response': result['text'].strip(),
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
            'total_tokens': result['total_tokens']
        }
    except Exception as e:
        logger.exception("Error processing follow-up question: %s", e)
        return {
            'response': f"Error processing follow-up question: {str(e)}",
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }


def generate_sql_from_question(user_question, schema, conversation_memory=None):
    """Generates a SQLite-optimized query from natural language."""
    try:
        logger.info(f"Generating SQL for: {user_question}")
        
        # Optimized prompt structure
        prompt = f"""
        ### ROLE
        You are a specialized SQLite Expert. Your task is to convert user questions into syntactically correct SQLite SELECT statements based on the provided schema.

        ### SCHEMA
        - Table Name: {schema['table_name']}
        - Columns: {json.dumps(schema['columns'])}

        ### SQLITE DIALECT RULES
        1. OUTPUT FORMAT: Return ONLY a JSON object: {{"sql": "your_query"}}. No markdown, no backticks, no explanations.
        2. DATE FILTERING (EXACT): Use date('YYYY-MM-DD') for exact date matches and use strftime('%Y', column_name) = 'YYYY'.
        4. DATE FILTERING (RELATIVE): Use date('now', '-X days') or date('now', '-X months').
        5. PRIMARY DATE COLUMN: Strictly Always use `Open_date` for general date-related questions unless another column is specified.
        6. DATE MATH: Use (julianday(date1) - julianday(date2)) to calculate differences in days.
        7. STRING MATCHING: Use the `LIKE` operator for case-insensitive searches.
        8. CATEGORICAL VALUES: 
           - 'consumer_type' values: (SME, Corporate, Individual)
           - 'record_type' values: (New, Existing, Updated)
        9. NULL CASE: If the request is not a query or cannot be answered, return {{"sql": ""}}.
        10. Always return the relevant columns  as per user question in the SELECT statement, do not use SELECT *.
        ### FEW-SHOT EXAMPLES
        - User: "341 meeting exactly 30 days after filing"
          Assistant: {{"sql": "SELECT * FROM {schema['table_name']} WHERE CAST(julianday(\"341_date\") - julianday(date_filed) AS INTEGER) = 30"}}
        
        - User: "Filings from 2023"
          Assistant: {{"sql": "SELECT * FROM {schema['table_name']} WHERE strftime('%Y', Open_date) = '2023'"}}

        ### USER REQUEST
        "{user_question}"
        """

        result = call_llm_with_tokens(prompt)
        sql_query = extract_sql_from_response(result['text'])

        return {
            'sql_query': sql_query.rstrip(';'),
            'input_tokens': result.get('input_tokens', 0),
            'output_tokens': result.get('output_tokens', 0),
            'total_tokens': result.get('total_tokens', 0)
        }

    except Exception as e:
        logger.error(f"SQL Generation failed: {e}", exc_info=True)
        return {'sql_query': '', 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}



def validate_sql_with_judge(sql_query, user_question, schema):
    """Use an LLM as a judge to validate if the generated SQL is correct.
    Returns: {
        'is_valid': bool,
        'status': str,
        'explanation': str,
        'repaired_query': str or None,
        'input_tokens': int,
        'output_tokens': int,
        'total_tokens': int
    }
    """
    
    # Step 1: Check for DDL and DML queries - BLOCK these operations
    sql_upper = sql_query.strip().upper()
    
    # DDL (Data Definition Language) keywords - CREATE, ALTER, DROP, TRUNCATE, RENAME
    ddl_keywords = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'RENAME']
    # DML (Data Manipulation Language) keywords - INSERT, UPDATE, DELETE, MERGE
    dml_keywords = ['INSERT', 'UPDATE', 'DELETE', 'MERGE']
    
    # Check if the query starts with or contains DDL/DML keywords
    for keyword in ddl_keywords + dml_keywords:
        if sql_upper.startswith(keyword) or re.search(r'\b' + keyword + r'\b', sql_upper):
            operation_type = "DDL" if keyword in ddl_keywords else "DML"
            logger.warning("Blocked %s operation: %s", operation_type, sql_query)
            return {
                "is_valid": False,
                "status": "BLOCKED",
                "explanation": f"{operation_type} queries are not allowed. Only SELECT queries are permitted.",
                "repaired_query": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
    
    # Step 2: Validate that query is a SELECT query
    if not sql_upper.startswith('SELECT'):
        logger.warning("Non-SELECT query attempted: %s", sql_query)
        return {
            "is_valid": False,
            "status": "INVALID",
            "explanation": "Only SELECT queries are allowed. The query must start with SELECT.",
            "repaired_query": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    
    # Step 3: Use LLM to validate column names, syntax, and correctness
    try:
        columns_desc = "\n".join([
            f"- {col['name']} ({col['type']}): {col['description']}"
            for col in schema['columns']
        ])
        
        prompt = f"""You are an expert SQLite3 database validator. Your task is to judge whether the following SQL query is correct and valid. And treat each row in the table as one backstop.

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
7. Reject any queries that attempt to modify data (INSERT, UPDATE, DELETE are already blocked)

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT * FROM {schema['table_name']} WHERE severity = 'High'"}}

Do not include any explanations, text, or additional content outside the JSON object."""

        result = call_llm_with_tokens(prompt)
        response = result['text']
        logger.info("SQL Validation Response:\n%s", response)
        
        # Parse the response robustly
        is_valid = False
        explanation = "Validation response received"
        query = None
        
        response_text = response.strip()
        
        if 'YES' in response_text.upper():
            explanation = "Query is valid"
            is_valid = True
        else:
            explanation = "❌ Query is invalid"
            query = extract_sql_from_response(response_text)
        
        return {
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID",
            "explanation": explanation,
            "repaired_query": query if not is_valid else None,
            "input_tokens": result['input_tokens'],
            "output_tokens": result['output_tokens'],
            "total_tokens": result['total_tokens'],
        }
    except Exception as e:
        logger.exception("Error validating SQL with judge: %s", e)
        return {
            "is_valid": False,
            "status": "ERROR",
            "explanation": f"❌ Validation error: {str(e)}",
            "suggestion": None,
            "repaired_query": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

def clean_sql_for_whitespace(sql_query):
    """Add TRIM() to WHERE clauses to handle whitespace in data"""
    try:
        import re
        
        # Pattern to match WHERE conditions with string values
        # Matches: column_name = 'value' or column_name = "value"
        pattern = r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'"])([^\2]*?)\2'
        
        def replace_with_trim(match):
            col_name = match.group(1)
            quote = match.group(2)
            value = match.group(3)
            # Return the same but with TRIM() wrapped around column
            return f"WHERE TRIM({col_name}) = {quote}{value}{quote}"
        
        modified_query = re.sub(pattern, replace_with_trim, sql_query, flags=re.IGNORECASE)
        
        if modified_query != sql_query:
            logger.info("Added TRIM() to WHERE clause for whitespace handling | original_length=%d | modified_length=%d", len(sql_query), len(modified_query))
        
        return modified_query
    except Exception as e:
        logger.exception("Error cleaning SQL for whitespace: %s", e)
        return sql_query


def execute_sql_query(sql_query, table_name):
    """Execute SQL query against the database and return results"""
    try:
        # Log the original SQL for debugging
        logger.info("Original SQL query: %s", sql_query)
        
        # Clean up SQL for whitespace issues
        sql_query = clean_sql_for_whitespace(sql_query)
        # st.markdown(f"**Generated SQL:** {sql_query}")
        logger.info("Executing SQL query (after cleanup): %s", sql_query)
        
        # Validate that query references the correct table
        if table_name.lower() not in sql_query.lower():
            logger.warning("Generated SQL doesn't reference the correct table. Adding FROM clause.")

        # Get actual columns from database to validate the query
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(bankruptcy4);")
        db_columns = [row[1] for row in cursor.fetchall()]
        
        # Note: We skip column validation for aggregation queries (COUNT, SUM, etc)
        # as they don't reference actual columns in the same way
        is_aggregation_query = any(keyword in sql_query.upper() for keyword in ['COUNT(*)', 'COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
        
        if not is_aggregation_query:
            # Extract column names from the SQL query for validation
            import re
            
            # Look for column references in SELECT clause and WHERE/AND/OR conditions
            select_columns = re.findall(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
            where_columns = re.findall(r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]', sql_query)
            
            potential_columns = []
            
            # Extract from SELECT clause (if found)
            if select_columns:
                # Split by comma and clean up each column
                cols = select_columns[0].split(',')
                for col in cols:
                    col = col.strip()
                    # Extract column name (handle aliases like "col AS alias")
                    col_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', col)
                    if col_match:
                        potential_columns.append(col_match.group(1))
            
            # Add WHERE clause columns
            potential_columns.extend(where_columns)
            potential_columns = list(set(potential_columns))
            
            # Table names, common aliases, and placeholder words to exclude
            exclude_names = {'bankruptcy4', 'data', 'table', 'db', 'a', 'b', 'c', 't', 'u', 'd', 'Column', 'column', 'name', 'value', 'result', 'row', 'query', 'COUNT', 'Sum', 'Count', 'Average'}
            
            # Check for SQL keywords that might be in the regex results
            sql_keywords = {'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'LIMIT', 'ORDER', 'BY', 'ASC', 'DESC', 'GROUP', 'HAVING', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CAST', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON', 'JOIN', 'LEFT', 'INNER', 'OUTER', 'CROSS', 'LIKE', 'IN', 'IS', 'NOT', 'NULL', 'BETWEEN', 'INSERT', 'UPDATE', 'DELETE'}
            
            # Filter out SQL keywords and excluded names
            referenced_columns = [col for col in potential_columns if col not in sql_keywords and col not in exclude_names and col.upper() not in sql_keywords]
            
            # Check if all referenced columns exist in the database
            missing_columns = [col for col in referenced_columns if col not in db_columns]
            
            if missing_columns:
                error_msg = f"Column(s) {missing_columns} do not exist in the database. Available columns: {db_columns}"
                logger.error(error_msg)
                conn.close()
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border: 2px solid #fca5a5;
                        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
                    ">
                        <p style="margin: 0; color: #7f1d1d; font-weight: 600;">❌ Column Validation Error</p>
                        <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.9rem;">The following columns do not exist: <strong>{', '.join(missing_columns)}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return None
        
        # Execute query
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        logger.info("SQL query executed successfully | rows=%d", len(df))
        
        # If query returned 0 rows, show debugging info
        if len(df) == 0:
            logger.warning("Query returned 0 rows. Showing sample data and debug info...")
            try:
                conn = sqlite3.connect('data.db')
                # Show sample data
                sample_df = pd.read_sql_query("SELECT * FROM bankruptcy4 LIMIT 5;", conn)
                logger.info("Sample data from database:\n%s", sample_df)
                
                # Extract WHERE condition to help debug
                import re
                where_match = re.search(r'WHERE\s+(.+?)(?:LIMIT|;|$)', sql_query, re.IGNORECASE)
                if where_match:
                    where_clause = where_match.group(1).strip()
                    # Try to extract column name and value
                    col_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*[\'"]?([^\'"]+)[\'"]?', where_clause)
                    if col_match:
                        col_name = col_match.group(1)
                        col_value = col_match.group(2)
                        logger.info("WHERE clause analysis: column='%s', value='%s'", col_name, col_value)
                        
                        # Show distinct values for this column
                        try:
                            distinct_df = pd.read_sql_query(f"SELECT DISTINCT {col_name} FROM bankruptcy4 LIMIT 10;", conn)
                            logger.info("Sample distinct values for '%s':\n%s", col_name, distinct_df)
                        except Exception as e:
                            logger.warning("Could not fetch distinct values: %s", e)
                
                conn.close()
            except Exception as e:
                logger.warning("Error during debug analysis: %s", e)
        
        return df
    except Exception as e:
        logger.exception("Error executing SQL query: %s", e)
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                padding: 1.25rem;
                border-radius: 10px;
                border: 2px solid #fca5a5;
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
            ">
                <p style="margin: 0; color: #7f1d1d; font-weight: 600;">❌ SQL Execution Error</p>
                <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.9rem;">An error occurred while executing the query.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None
    
def should_generate_insights(user_query, result_df):
    """Generate insights only when user intent explicitly asks for it."""
    if result_df is None or result_df.empty or len(result_df) < 2:
        logger.debug("Skipping insights generation: empty or insufficient data | rows=%d", len(result_df) if result_df is not None else 0)
        return False

    insight_keywords = {
        "insight",
        "insights",
        "plot",
        "plots",
        "chart",
        "charts",
        "graph",
        "generate insights",
        "visualization",
        "draw",
        "graphs",
        "visual",
        "visualize",
        "visualise",
        "visualization",
        "distribution",
        "trend",
        "breakdown",
        "compare",
        "comparison",
        "percentage",
        "share",
        "dashboard",
        "summary",
    }
    query_lower = user_query.lower()
    should_generate = any(keyword in query_lower for keyword in insight_keywords)
    logger.debug("Insights generation check | should_generate=%s | keywords_found=%s", should_generate, 
                 [k for k in insight_keywords if k in query_lower])
    return should_generate

def _initialize_conversation_memory():
    """Initialize conversation memory structure"""
    memory = {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
    }
    logger.info("Conversation memory initialized")
    return memory


def _display_token_info(tokens_dict: dict):
    """Display token usage information with an info button."""
    try:
        if not tokens_dict:
            return
        
        # Calculate totals
        total_input = 0
        total_output = 0
        total_all = 0
        
        # Handle both single token dicts and nested token structure
        if 'input_tokens' in tokens_dict:
            total_input = tokens_dict.get('input_tokens', 0)
            total_output = tokens_dict.get('output_tokens', 0)
            total_all = tokens_dict.get('total_tokens', 0)
        elif 'sql_generation' in tokens_dict or 'sql_validation' in tokens_dict:
            sql_gen = tokens_dict.get('sql_generation', {})
            sql_val = tokens_dict.get('sql_validation', {})
            answer_gen = tokens_dict.get('answer_generation', {})
            
            total_input = sql_gen.get('input_tokens', 0) + sql_val.get('input_tokens', 0) + answer_gen.get('input_tokens', 0)
            total_output = sql_gen.get('output_tokens', 0) + sql_val.get('output_tokens', 0) + answer_gen.get('output_tokens', 0)
            total_all = total_input + total_output
        
        if total_all > 0:
            col1, col2 = st.columns([0.88, 0.12])
            with col2:
                help_text = f"📊 Token Usage\n\nInput Tokens: {total_input:,}\nOutput Tokens: {total_output:,}\nTotal Tokens: {total_all:,}"
                st.popover(
                    "ℹ️ ",
                    help=help_text
                )
    except Exception as e:
        logger.debug("Error displaying token info: %s", e)


def _append_conversation_memory(user_query: str, sql_query: str = "", records: list = None, validation_result: dict = None, answer: str = None, 
                                 sql_tokens: dict = None, validation_tokens: dict = None, answer_tokens: dict = None):
    """
    Append structured conversation memory with user question, SQL query, records, and token counts.
    
    Args:
        user_query: The user's natural language question
        sql_query: The generated SQL query
        records: List of dictionaries containing fetched records
        validation_result: Optional validation result dictionary
        answer: The assistant's response to the user's query
        sql_tokens: Token counts from SQL generation {'input_tokens': int, 'output_tokens': int, 'total_tokens': int}
        validation_tokens: Token counts from SQL validation
        answer_tokens: Token counts from answer generation
    """
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory
        
        # Build structured memory entry
        if not answer:
            memory_entry = {
                "user_question": user_query,
                "sql_query": sql_query,
                "records": records or [],
                "record_count": len(records) if records else 0,
                "tokens": {
                    "sql_generation": sql_tokens or {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                    "sql_validation": validation_tokens or {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                }
            }
        else:
            memory_entry = {
                "user_question": user_query,
                "assistant_answer": answer,
                "tokens": {
                    "answer_generation": answer_tokens or {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                }
            }
        
        # Add validation result if provided
        if validation_result:
            memory_entry["validation_result"] = validation_result.get("status", "UNKNOWN")
        
        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query
        if not answer:
            memory["last_assistant_response"] = f"SQL Query: {sql_query} | Total Number of Records: {len(records) if records else 0} | Records: {records}"
        else:
            memory["last_assistant_response"] = answer

        logger.info("Conversation memory appended | query_length=%d | records=%d | total_tokens=%d | history_length=%d", 
                   len(user_query), len(records) if records else 0, 
                   (sql_tokens or {}).get('total_tokens', 0) + (validation_tokens or {}).get('total_tokens', 0),
                   len(memory["history"]))

        # Keep only last 10 conversations
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]
            logger.debug("Conversation memory trimmed to last 10 entries")
    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)


def _build_memory_context() -> str:
    """Build context string from conversation memory for LLM context."""
    try:
        memory = st.session_state.get("conversation_memory")
        if not memory or not memory.get("history"):
            logger.debug("No conversation history found")
            return "No prior conversation context."
        logger.debug("Memory context built from %d conversation entries", len(memory.get("history", [])))
        return memory
    except Exception as e:
        logger.exception("Error building memory context: %s", e)
        return "No prior conversation context."

def _load_and_clean_csv(uploaded_file):
    """Load and clean CSV data - handles whitespace in column names and data"""
    try:
        df = pd.read_csv(uploaded_file)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        # Remove columns with empty names (columns that were only whitespace)
        df = df.loc[:, df.columns != '']
        
        # Clean data: strip whitespace from all string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        logger.info("CSV loaded and cleaned | filename=%s | shape=%s", uploaded_file.name, df.shape)
        return df
    except Exception as e:
        logger.exception("Error loading and cleaning CSV: %s", e)
        return None
    
def session_changer():
    """Function to trigger re-rendering of the chat when follow-up questions toggle is changed."""
    follow_up_enabled = st.session_state.follow_up_toggle_state
    st.session_state.follow_up_toggle = follow_up_enabled
    logger.info("Follow-up questions toggle changed | enabled=%s", follow_up_enabled)

def main():
    logger.info("=== Streamlit session started ===")
    
    # Initialize conversation memory if not present
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = _initialize_conversation_memory()
    
    # Load schema
    logger.info("Loading database schema")
    schema = load_schema()
    if not schema:
        logger.error("Failed to load schema - schema.json missing")
        st.error("Failed to load database schema. Please ensure schema.json exists.")
        return

    # Sidebar
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
            padding: 1.5rem;
            border-radius: 14px;
            border: 1px solid #d1f2f7;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
        ">
            <h2 style="
                margin: 0 0 1rem 0;
                color: #0284c7;
                font-size: 1.3rem;
                font-weight: 700;
                letter-spacing: -0.3px;
            ">📁 Data Management</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Initialize session state for tracking uploaded file
    if "last_uploaded_file_name" not in st.session_state:
        st.session_state.last_uploaded_file_name = None
    if "data_in_db" not in st.session_state:
        st.session_state.data_in_db = False
    if "actual_schema" not in st.session_state:
        st.session_state.actual_schema = None
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "follow_up_toggle" not in st.session_state:
        st.session_state.follow_up_toggle = False

    if uploaded_file is not None:
        # Check if this is a new file
        file_changed = uploaded_file.name != st.session_state.last_uploaded_file_name
        
        # Load and clean CSV data
        df = _load_and_clean_csv(uploaded_file)
        if df is None:
            logger.error("Failed to load CSV file: %s", uploaded_file.name)
            st.error(f"Could not process the uploaded file.")
            st.session_state.data_in_db = False
            return
        
        if file_changed:
            logger.info("New file detected | filename=%s", uploaded_file.name)
            try:
                st.sidebar.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border-left: 4px solid #10b981;
                        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
                        margin-bottom: 1rem;
                    ">
                        <p style="margin: 0; color: #065f46; font-weight: 600;">✅ File loaded successfully!</p>
                        <p style="margin: 0.5rem 0 0 0; color: #047857; font-size: 0.9rem;"><strong>{uploaded_file.name}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.sidebar.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                        padding: 1rem;
                        border-radius: 10px;
                        border: 1px solid #d1f2f7;
                        margin-bottom: 1rem;
                    ">
                        <p style="margin: 0; color: #0284c7; font-weight: 600; font-size: 0.95rem;">Data Preview</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.sidebar.dataframe(df.head(500), hide_index=True, use_container_width=True)
                
                # Auto-save to SQLite
                conn = sqlite3.connect('data.db')
                df.to_sql('bankruptcy4', conn, if_exists='replace', index=False)
                conn.close()
                
                # Update session state
                st.session_state.last_uploaded_file_name = uploaded_file.name
                st.session_state.data_in_db = True
                
                # Get actual database schema
                st.session_state.actual_schema = get_actual_database_schema()
                st.sidebar.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border-left: 4px solid #10b981;
                        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
                        margin-bottom: 1rem;
                    ">
                        <p style="margin: 0; color: #065f46; font-weight: 600;">✅ Database Updated</p>
                        <p style="margin: 0.5rem 0 0 0; color: #047857; font-size: 0.9rem;"><strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                logger.info("Data saved to SQLite | filename=%s | rows=%d | columns=%d", uploaded_file.name, df.shape[0], df.shape[1])
            except Exception as e:
                logger.exception("Failed to process and save uploaded file: %s", uploaded_file.name)
                st.error(f"Could not process the uploaded file. (Detail: {e})")
                st.session_state.data_in_db = False
                return
        else:
            logger.debug("Same file already loaded | filename=%s", uploaded_file.name)
            # Same file, show cached data
            st.sidebar.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                    padding: 1.25rem;
                    border-radius: 10px;
                    border-left: 4px solid #0284c7;
                    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
                    margin-bottom: 1rem;
                ">
                    <p style="margin: 0; color: #0c4a6e; font-weight: 600;">📄 File Ready</p>
                    <p style="margin: 0.5rem 0 0 0; color: #0369a1; font-size: 0.9rem;"><strong>{uploaded_file.name}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.sidebar.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                    padding: 1rem;
                    border-radius: 10px;
                    border: 1px solid #d1f2f7;
                    margin-bottom: 1rem;
                ">
                    <p style="margin: 0; color: #0284c7; font-weight: 600; font-size: 0.95rem;">Data Preview</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.sidebar.dataframe(df.head(500), hide_index=True, use_container_width=True)

    # Check if data exists in database
    if st.session_state.data_in_db:
        try:
            conn = sqlite3.connect('data.db')
            check_df = pd.read_sql_query("SELECT COUNT(*) as count FROM bankruptcy4;", conn)
            conn.close()
            count = check_df['count'].values[0] if len(check_df) > 0 else 0
            if count > 0:
                logger.info("Database contains %d records", count)
                st.sidebar.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border: 1px solid #d1fae5;
                        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
                        margin-bottom: 1rem;
                    ">
                        <p style="margin: 0; color: #065f46; font-weight: 700; font-size: 1.1rem;">✨ {count}</p>
                        <p style="margin: 0.5rem 0 0 0; color: #047857; font-weight: 500;">Records Available</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Show detected columns
                if st.session_state.actual_schema:
                    col_names = [col['name'] for col in st.session_state.actual_schema['columns']]
                    logger.debug("Detected %d columns in schema", len(col_names))
                    st.sidebar.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                            padding: 1rem;
                            border-radius: 10px;
                            border: 1px solid #d1f2f7;
                            margin-bottom: 1rem;
                        ">
                            <p style="margin: 0; color: #0284c7; font-weight: 600; font-size: 0.95rem;">🔍 Detected Columns: <strong>{len(col_names)}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.sidebar.expander("📋 View all columns", expanded=False):
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                                padding: 1rem;
                                border-radius: 8px;
                                border: 1px solid #d1f2f7;
                            ">
                                <p style="margin: 0; color: #0c4a6e; font-size: 0.9rem; line-height: 1.8;">
                                    {chr(10).join([f"<strong>•</strong> {col}" for col in col_names])}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                logger.warning("Database is empty - no records found")
                st.session_state.data_in_db = False
        except Exception as e:
            logger.debug("Database check failed: %s", e)
            st.session_state.data_in_db = False

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello ! I can help you query your bankruptcy4 database using natural language. Just ask me any question about your data!",
            }
        ]

    # Replay previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display dataframe if present
            if msg["role"] == "assistant":
                # if "sql_query" in msg:
                #     with st.expander("Generated SQL"):
                #         st.code(msg["sql_query"], language="sql")
                        
                # if "validation_result" in msg:
                #     validation_result = msg["validation_result"]
                #     with st.expander("SQL Validation Report"):
                #         st.markdown(f"**Status:** `{validation_result['status']}`")
                #         st.markdown(f"**Explanation:** {validation_result['explanation']}")
                #         if validation_result.get('suggestion'):
                #             st.markdown(f"**Suggested Query:** ")
                #             st.code(validation_result['suggestion'], language="sql")
                            
                #         # If invalid, check if there was a repaired query
                #         if not validation_result.get('is_valid') and validation_result.get('repaired_query'):
                #             st.warning(f"SQL validation failed: {validation_result['explanation']}")
                #             st.success("Auto-repaired SQL:")
                #             st.code(validation_result['repaired_query'], language="sql")
                            
                # Display dataframe if present
                if "dataframe" in msg and not msg.get("hide_dataframe", False):
                    st.dataframe(msg["dataframe"], width='content', hide_index=True)
                
                # Re-render insights/charts if they were originally generated
                if msg.get("has_insights") and msg.get("user_query"):
                    chart_type = msg.get("chart_type", "auto")
                    generate_insights(msg["dataframe"], chart_type=chart_type, user_query=msg["user_query"])

    # Chat input and follow-up toggle
    st.sidebar.markdown(
        """
        <hr style="border-color: #d1f2f7; margin: 1.5rem 0;">
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #d1f2f7;
            margin-bottom: 1rem;
        ">
            <p style="margin: 0 0 0.75rem 0; color: #0284c7; font-weight: 700; font-size: 0.95rem;">⚙️ Query Mode</p>
            <p style="margin: 0; color: #0c4a6e; font-size: 0.85rem; line-height: 1.5;">Enable follow-up questions to ask about previous results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.toggle("Follow-up questions", key="follow_up_toggle_state", on_change = session_changer)
    
    user_query = st.chat_input("💬 Ask me anything about your data...")
    if not user_query:
        return

    # Check if we have data in database before processing query
    if not st.session_state.data_in_db:
        logger.warning("Query attempted without data | query=%s", user_query)
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border: 2px solid #fca5a5;
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
                text-align: center;
            ">
                <p style="margin: 0; color: #7f1d1d; font-weight: 700; font-size: 1.1rem;">📁 No Data Available</p>
                <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.95rem;">Please upload a CSV file first to query the database.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    logger.info("User query received | follow_up_mode=%s | query=%s", st.session_state.follow_up_toggle, user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Check if follow-up questions toggle is enabled
        if st.session_state.follow_up_toggle:
            logger.info("Processing as follow-up question")
            with st.spinner("Generating follow-up response..."):
                result = follow_up_question(user_query)
                response = result['response']
                answer_tokens = {
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'total_tokens': result['total_tokens']
                }
                
                st.markdown(response)
                _display_token_info(answer_tokens)
                
                # Fetch last dataframe to generate graph if requested
                last_df = None
                for msg in reversed(st.session_state.messages):
                    if "dataframe" in msg:
                        last_df = msg["dataframe"]
                        break
                
                _has_insights = False
                chart_type = None
                if last_df is not None and not last_df.empty and should_generate_insights(user_query, last_df):
                    chart_type = detect_chart_type(user_query)
                    st.info(f"Generating {chart_type if chart_type != 'auto' else 'AI-driven'} chart based on your request...")
                    generate_insights(last_df, chart_type=chart_type, user_query=user_query)
                    _has_insights = True
                
                assistant_entry = {"role": "assistant", "content": response, "tokens": answer_tokens}
                if _has_insights:
                    assistant_entry.update({
                        "has_insights": True, "chart_type": chart_type, 
                        "dataframe": last_df, "hide_dataframe": True, "user_query": user_query
                    })
                st.session_state.messages.append(assistant_entry)
            _append_conversation_memory(user_query=user_query, answer=response, sql_query="", records=[], answer_tokens=answer_tokens)
        else:
            logger.info("Processing as standard query")
            with st.spinner("Analyzing your question and generating response..."):
                _handle_user_query(user_query, schema)

def detect_chart_type(user_query):
    """Detect whether the user explicitly asked for a bar chart or pie chart."""
    try:
        query_lower = user_query.lower()

        pie_keywords = ["pie chart", "piechart", "pie", "donut"]
        bar_keywords = ["bar chart", "barchart", "bar plot", "barplot", "histogram", "column chart", "columnchart", "bar"]

        if any(keyword in query_lower for keyword in pie_keywords):
            logger.debug("Chart type detected: PIE")
            return "pie"
        if any(keyword in query_lower for keyword in bar_keywords):
            logger.debug("Chart type detected: BAR")
            return "bar"
        logger.debug("Chart type detected: AUTO")
        return "auto"
    except Exception as e:
        logger.exception("Error detecting chart type: %s", e)
        return "auto"

def _handle_user_query(user_query, schema):
    """Handle user queries by generating, validating, and executing SQL"""
    try:
        cache_key = user_query.strip().lower()
        
        # Check cache first
        if cache_key in st.session_state.query_cache:
            logger.info("Query found in cache | query=%s", user_query)
            cached_data = st.session_state.query_cache[cache_key]
            
            sql_query = cached_data["sql_query"]
            result_df = cached_data["result_df"]
            
            # Convert dataframe to records for memory
            records = result_df.to_dict(orient='records') if len(result_df) > 0 else []
            _append_conversation_memory(user_query, sql_query, records)
            
            # with st.expander("SQL Query"):
            #     st.code(sql_query, language="sql")
            if len(result_df) == 0:
                msg = "No records matched your criteria."
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border: 2px solid #fbbf24;
                        box-shadow: 0 4px 12px rgba(251, 146, 60, 0.15);
                    ">
                        <p style="margin: 0; color: #78350f; font-weight: 600;">⚠️ {msg}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({"role": "assistant", "content": msg})
                logger.info("Cached query returned 0 rows")
                return
                
            success_msg = f"✅ Found {len(result_df)} records (from cache)"
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                    padding: 1.25rem;
                    border-radius: 10px;
                    border: 2px solid #10b981;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
                ">
                    <p style="margin: 0; color: #065f46; font-weight: 600;">⚡ Results Retrieved from Cache</p>
                    <p style="margin: 0.5rem 0 0 0; color: #047857; font-size: 0.9rem;"><strong>{len(result_df)} records</strong> found matching your criteria.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(result_df, width='content', hide_index=True)
            
            if should_generate_insights(user_query, result_df):
                chart_type = detect_chart_type(user_query)
                # if chart_type != "auto":
                #     st.info(f"Generating insights and {chart_type} chart based on your request...")
                # else:
                #     st.info("Generating AI-driven insights and visualizations from the result set...")
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
                
            _has_insights = should_generate_insights(user_query, result_df)
            assistant_entry = {
                "role": "assistant",
                "content": success_msg,
                "sql_query": sql_query,
                "dataframe": result_df,
                "user_query": user_query,
                "has_insights": _has_insights,
                "chart_type": detect_chart_type(user_query) if _has_insights else None
            }
            st.session_state.messages.append(assistant_entry)
            logger.info("Cached query executed successfully | rows=%d", len(result_df))
            return


        # Step 1: Generate SQL from natural language question
        logger.info("Starting SQL generation workflow | query=%s", user_query)
        #st.info("Step 1: Generating SQL from your question...")
        sql_result = generate_sql_from_question(user_query, schema)
        sql_query = sql_result['sql_query']
        sql_tokens = {
            'input_tokens': sql_result['input_tokens'],
            'output_tokens': sql_result['output_tokens'],
            'total_tokens': sql_result['total_tokens']
        }
        
        if not sql_query:
            error_msg = "Failed to generate SQL query from your question. Please try rephrasing."
            logger.warning("SQL generation failed for query: %s", user_query)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                    padding: 1.25rem;
                    border-radius: 10px;
                    border: 2px solid #fca5a5;
                    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
                ">
                    <p style="margin: 0; color: #7f1d1d; font-weight: 600;">❌ {error_msg}</p>
                    <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.9rem;">Try rephrasing your question with more specific details.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            _append_conversation_memory(user_query, "", [], None, sql_tokens=sql_tokens)
            return
        
        # Display generated SQL
        # with st.expander("Generated SQL"):
        #     st.code(sql_query, language="sql")
        
        # Step 2: Validate SQL with judge LLM
        logger.info("Validating generated SQL")
        #st.info("Step 2: Validating SQL with AI judge...")
        validation_result = validate_sql_with_judge(sql_query, user_query, schema)
        validation_tokens = {
            'input_tokens': validation_result['input_tokens'],
            'output_tokens': validation_result['output_tokens'],
            'total_tokens': validation_result['total_tokens']
        }
        
        # Display validation result
        # with st.expander("SQL Validation Report"):
        #     st.markdown(f"**Status:** `{validation_result['status']}`")
        #     st.markdown(f"**Explanation:** {validation_result['explanation']}")
        
        # Check if validation passed and use appropriate query
        final_query = sql_query
        
        if not validation_result['is_valid']:
            # Auto-repair was attempted
            if validation_result.get('repaired_query'):
                logger.warning("SQL validation failed, attempting auto-repair")
                with st.expander("🔧 View Repaired Query"):
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #f0fdfd 0%, #ecf1f7 100%);
                            padding: 1rem;
                            border-radius: 8px;
                            border: 1px solid #d1f2f7;
                        ">
                            <p style="margin: 0 0 0.5rem 0; color: #0284c7; font-size: 0.85rem; font-weight: 600;">Auto-Repaired SQL Query:</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(validation_result['repaired_query'], language="sql")
                final_query = validation_result['repaired_query']
            else:
                error_msg = f"SQL validation failed: {validation_result['explanation']}"
                logger.error("SQL validation failed with no repair possible: %s", error_msg)
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                        padding: 1.25rem;
                        border-radius: 10px;
                        border: 2px solid #fca5a5;
                        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
                    ">
                        <p style="margin: 0; color: #7f1d1d; font-weight: 600;">❌ Validation Failed</p>
                        <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.9rem;">{validation_result['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                _append_conversation_memory(user_query, sql_query, [], validation_result)
                return
        else:
            logger.info("SQL validation passed")
            # st.success("SQL validation passed!")
        
        # Step 3: Execute the SQL query
        logger.info("Executing SQL query")
        result_df = execute_sql_query(final_query, schema['table_name'])
        
        if result_df is None:
            error_msg = "Error executing the SQL query. Please check the database or try a different question."
            logger.error("SQL execution failed")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                    padding: 1.25rem;
                    border-radius: 10px;
                    border: 2px solid #fca5a5;
                    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
                ">
                    <p style="margin: 0; color: #7f1d1d; font-weight: 600;">❌ Execution Error</p>
                    <p style="margin: 0.75rem 0 0 0; color: #991b1b; font-size: 0.9rem;">The query could not be executed. Please try a different question.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            _append_conversation_memory(user_query, final_query, [], validation_result, sql_tokens=sql_tokens, validation_tokens=validation_tokens)
            return
        
        if len(result_df) == 0:
            msg = "Query executed successfully, but no records matched your criteria."
            logger.info("Query executed but returned 0 rows")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
                    padding: 1.25rem;
                    border-radius: 10px;
                    border: 2px solid #fbbf24;
                    box-shadow: 0 4px 12px rgba(251, 146, 60, 0.15);
                ">
                    <p style="margin: 0; color: #78350f; font-weight: 600;">⚠️ No Results</p>
                    <p style="margin: 0.75rem 0 0 0; color: #92400e; font-size: 0.9rem;">{msg}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "content": msg})
            _append_conversation_memory(user_query, final_query, [], validation_result, sql_tokens=sql_tokens, validation_tokens=validation_tokens)
            return
        
        # Display results
        success_msg = f"✅ Found {len(result_df)} records"
        logger.info("Query executed successfully | rows=%d", len(result_df))
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                padding: 1.25rem;
                border-radius: 10px;
                border: 2px solid #10b981;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
                margin-bottom: 1.5rem;
            ">
                <p style="margin: 0; color: #065f46; font-weight: 700; font-size: 1.05rem;">✨ Query Successful</p>
                <p style="margin: 0.5rem 0 0 0; color: #047857; font-size: 0.9rem;"><strong>{len(result_df)} records</strong> found related to your question.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(result_df, width='content', hide_index=True)

        if should_generate_insights(user_query, result_df):
            chart_type = detect_chart_type(user_query)
            if chart_type != "auto":
                logger.info("Generating insights with %s chart", chart_type)
                #st.info(f"Step 4: Generating insights and {chart_type} chart based on your request...")
            else:
                logger.info("Generating auto-detected insights")
                #st.info("Step 4: Generating AI-driven insights and visualizations from the result set...")
            generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        _has_insights = should_generate_insights(user_query, result_df)
        
        # Convert dataframe to records for memory storage
        records = result_df.to_dict(orient='records')
        
        # Display token info with info button
        col1, col2 = st.columns([0.95, 0.05])
        with col2:
            token_summary = f"SQL Gen: {sql_tokens['total_tokens']} | Validation: {validation_tokens['total_tokens']}"
            st.popover("ℹ️", help=token_summary)
        
        assistant_entry = {
            "role": "assistant",
            "content": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "user_query": user_query,
            "has_insights": _has_insights,
            "chart_type": detect_chart_type(user_query) if _has_insights else None,
            "tokens": {
                "sql_generation": sql_tokens,
                "sql_validation": validation_tokens
            }
        }
        st.session_state.messages.append(assistant_entry)
        
        # Cache the query with token info
        cache_key = user_query.strip().lower()
        st.session_state.query_cache[cache_key] = {
            "sql_query": final_query,
            "result_df": result_df,
            "tokens": assistant_entry["tokens"]
        }
        
        # Append conversation memory once with complete data including tokens
        _append_conversation_memory(user_query, final_query, records, validation_result, 
                                   sql_tokens=sql_tokens, validation_tokens=validation_tokens)
        
        # Save to cache
        st.session_state.query_cache[user_query.strip().lower()] = {
            "sql_query": final_query,
            "result_df": result_df
        }
        logger.info("Query execution completed and cached | rows=%d | cache_size=%d", len(result_df), len(st.session_state.query_cache))
        
    except Exception as error:
        error_msg = f"Unexpected error: {str(error)}"
        logger.exception("Unexpected error in query handling: %s", error_msg)
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    try:
        logger.info("Starting bankruptcy4 GenBI application")
        main()
    except Exception as e:
        logger.exception("Fatal error in main application: %s", e)
        raise