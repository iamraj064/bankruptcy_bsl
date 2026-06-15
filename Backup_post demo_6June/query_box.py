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
from insights_generator import generate_insights

# =============================================================================
# CONFIG & INITIALIZATION
# =============================================================================
load_dotenv()

logger = logging.getLogger("bankruptcy_genbi")

# =============================================================================
# SCHEMA UTILITY & FORMATTING
# =============================================================================

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
            schema["columns"].append({
                "name": col_name,
                "type": col_type,
                "description": descriptions.get(col_name.lower(), f"Database field: {col_name}")
            })
        return schema
    except Exception as e:
        logger.error(f"Error getting dynamic database schema: {e}")
        return None


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
            f"- {col.get('name') or col.get('Column Name')} ({col.get('type') or col.get('Data Type')}): {col.get('description') or col.get('Description')}"
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


def _make_query_cache_key(user_query, schema, conversation_memory=None):
    """Create a stable cache key for repeated user queries."""
    try:
        schema_str = json.dumps(schema, sort_keys=True)
        memory_str = json.dumps(conversation_memory or {}, sort_keys=True)
        combined = f"{user_query.strip().lower()}||{schema_str}||{memory_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    except Exception:
        return hashlib.sha256(user_query.strip().lower().encode('utf-8')).hexdigest()


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
        '>', '<', '=', '>=', '<='
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


def is_followup_question(user_query, conversation_history):
    """Detect if the user is asking a follow-up question."""
    if not st.session_state.temp_table_name:
        return False
    
    # 1. If it's a chart request and we have active data, treat as a follow-up
    if is_chart_request(user_query):
        return True
    
    # 2. Check standard follow-up keywords
    followup_keywords = ['that', 'those', 'these', 'records', 'results', 'data', 'rows', 'items', 'from that', 'from there', 'among those', 'above', 'convert', 'previous']
    query_lower = user_query.lower()
    return any(kw in query_lower for kw in followup_keywords)


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
# USER QUERY HANDLER & CHAT INTERACTION
# =============================================================================

def _handle_user_query(user_query, schema, active_schema):
    """Handle user queries by generating, validating, executing SQL and generating insights"""
    try:
        logger.info("Processing user query: %s", user_query)
        
        is_followup = is_followup_question(user_query, st.session_state.conversation_memory)
        
        if is_followup and st.session_state.temp_table_schema:
            working_schema = st.session_state.temp_table_schema
            st.info(" **Querying previous result set**")
            logger.info("Using temporary table for follow-up query | table=%s", working_schema.get('table_name'))
        else:
            working_schema = active_schema if active_schema else schema
            if any(kw in user_query.lower() for kw in ['clear', 'reset', 'new query', 'fresh', 'different']):
                if st.session_state.temp_table_name:
                    drop_temporary_table(st.session_state.temp_table_name)
                    st.session_state.temp_table_name = None
                    st.session_state.temp_table_schema = None
                    st.session_state.temp_table_source_query = None
                    st.session_state.temp_table_dataframe = None
                    st.info(" Cleared previous result set. Starting fresh...")
        
        generation_token_usage = {}
        validation_token_usage = {}
        token_usage = {
            "generation": generation_token_usage,
            "validation": validation_token_usage,
        }

        cache_key = _make_query_cache_key(user_query, working_schema, st.session_state.conversation_memory)

        if cache_key in st.session_state.query_cache:
            cached_response = st.session_state.query_cache[cache_key]
            st.success(" Cached result found for repeated query")
            st.dataframe(format_table(cached_response["dataframe"]), width='stretch', hide_index=True)
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
        if is_followup and st.session_state.temp_table_dataframe is not None and is_pure_chart_request(user_query):
            logger.info("Pure chart request detected. Bypassing SQL generation and reusing active dataset.")
            final_query = st.session_state.temp_table_source_query
            result_df = st.session_state.temp_table_dataframe
            validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
            is_visualization_fallback = True
            sql_query = None

        if not is_visualization_fallback:
            with st.spinner(""):
                sql_query = generate_sql_from_question(
                    user_query,
                    working_schema,
                    st.session_state.conversation_memory,
                    token_usage=generation_token_usage,
                )

            if not sql_query and is_followup and st.session_state.temp_table_dataframe is not None and is_chart_request(user_query):
                logger.info("Empty SQL generated for chart follow-up query. Re-using active dataset.")
                final_query = st.session_state.temp_table_source_query
                result_df = st.session_state.temp_table_dataframe
                validation_result = {"is_valid": True, "status": "VALID", "explanation": "Re-using active dataset for visualization"}
                is_visualization_fallback = True

        if not is_visualization_fallback:
            if not sql_query:
                error_msg = " Failed to generate SQL query from your question. Please try rephrasing."
                st.error(error_msg)
                _append_conversation_memory(user_query, "", None, answer=error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            
            with st.spinner(""):
                validation_result = validate_sql_with_judge(sql_query, user_query, working_schema)
            
            final_query = sql_query
            
            if not validation_result['is_valid']:
                if validation_result.get('repaired_query'):
                    st.warning(f" SQL validation indicated issues: {validation_result['explanation']}")
                    st.info(" Using auto-repaired query...")
                    final_query = validation_result['repaired_query']
                else:
                    error_msg = f" Sorry !!  {validation_result['explanation']}"
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
                st.success(" SQL validation passed!")
            
            with st.spinner(""):
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
            msg = " Query executed successfully, but no records matched your criteria."
            st.info(msg)
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
            st.session_state.query_cache[cache_key] = {
                "message": msg,
                "sql_query": final_query,
                "validation_result": validation_result,
                "dataframe": result_df,
                "records": [],
                "user_query": user_query,
                "has_insights": False,
                "chart_type": "auto",
                "token_usage": token_usage,
            }
            return
        
        success_msg = f" Generated chart for the active dataset:" if is_visualization_fallback else f" Query results: "
        st.success(success_msg)
        st.dataframe(format_table(result_df), width='stretch', hide_index=True)
        
        if not is_followup:
            temp_table_name, temp_schema = create_temporary_table_from_dataframe(result_df, final_query)
            if temp_table_name and temp_schema:
                st.session_state.temp_table_name = temp_table_name
                st.session_state.temp_table_schema = temp_schema
                st.session_state.temp_table_source_query = final_query
                st.session_state.temp_table_dataframe = result_df
                st.info(f" Result set stored as temporary table. You can ask follow-up questions about these {len(result_df)} records.")
        
        should_insights = should_generate_insights(user_query, result_df) or is_visualization_fallback
        chart_type = detect_chart_type(user_query)
        
        if should_insights and (len(result_df) >= 2 or is_visualization_fallback):
            with st.expander(" View Insights and Charts", expanded=True):
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        _append_conversation_memory(user_query, final_query, result_df.to_dict('records'), validation_result)
        
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
        st.session_state.query_cache[cache_key] = {
            "message": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "records": result_df.to_dict('records'),
            "user_query": user_query,
            "has_insights": should_insights,
            "chart_type": chart_type,
            "token_usage": token_usage,
        }
        
        logger.info("Query processed successfully | rows=%d | chart_type=%s | temp_table=%s", 
                    len(result_df), chart_type, st.session_state.temp_table_name or "none")
        
    except Exception as e:
        error_msg = f" Unexpected error: {str(e)}"
        logger.exception("Error in user query handling: %s", e)
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# =============================================================================
# WORKSPACE INTERFACE RENDERING
# =============================================================================

def render_query_box_tab(schema, active_schema):
    """Main rendering entrypoint for the Query Box tab in app.py"""
    st.subheader(" AI Transactional Assistant")
    st.markdown("Ask natural language queries about the filings database. The AI will translate it into a safe SQL query, validate it, execute it, and generate visual insights.")

    chat_box = st.container(height=500)

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
                        st.dataframe(format_table(df_res), width="stretch", hide_index=True)
                    elif content.get("type") == "text":
                        st.markdown(content.get("message", ""))
                    else:
                        st.write(content)
                else:
                    st.markdown(content)

                if msg["role"] == "assistant":
                    if "dataframe" in msg and msg["dataframe"] is not None:
                        st.dataframe(format_table(msg["dataframe"]), width="stretch", hide_index=True)

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

    st.divider()

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
                st.rerun()

    user_input = st.chat_input("Query database (e.g. 'Show total records by state')...")

    if user_input:
        if not st.session_state.data_in_db:
            st.error("No active dataset found in database. Please upload a CSV first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]
        query_text = last_msg["content"]

        with st.chat_message("assistant", avatar="blank.png"):
            with st.spinner("Processing request..."):
                _handle_user_query(query_text, schema, active_schema)
            st.rerun()