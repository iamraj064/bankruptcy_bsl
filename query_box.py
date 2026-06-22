import sqlite3
import logging
import sys
import json
import re
import hashlib
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


def load_schema():
    """Load default schema from schema.json"""
    try:
        with open("schema.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None


def clean_and_validate_suggestions(data, schema):
    """Post-process and validate suggestions to ensure they strictly conform to user rules."""
    visual_kws = ["show", "draw", "display", "plot", "chart", "graph", "visualize", "visualise", "breakdown", "distribution", "trend", "trends", "view", "pie", "bar", "line", "donut", "histogram", "heatmap", "scatter", "area"]
    
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
        
        # Check if it has any visual keyword
        has_visual_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in visual_kws)
        if has_visual_kw:
            # Try to rewrite it by replacing the visual action verb at the start
            cleaned_q = q_str
            cleaned_q = re.sub(r"^(show|display|draw|view|plot|visualize|visualise)\b", "", cleaned_q, flags=re.IGNORECASE).strip()
            q_lower_new = cleaned_q.lower()
            if any(re.search(rf"\b{kw}\b", q_lower_new) for kw in visual_kws):
                continue # discard it
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
        q_lower = q_str.lower()
        
        has_visual_kw = any(re.search(rf"\b{kw}\b", q_lower) for kw in visual_kws)
        if not has_visual_kw:
            # Prepend a default chart verb
            q_str = "Show " + q_str[0].lower() + q_str[1:]
        visual.append(q_str)
        
    # Deduplicate
    textual = list(dict.fromkeys(textual))
    visual = list(dict.fromkeys(visual))
    
    # 3. Formulate fallback lists for padding
    fallback_textual = [
        "Filter cases by Chapter 7 filings",
        "Identify the top 10 states by case volume",
        "List the most recent 10 records in the dataset",
        "Calculate the average match score for all records",
        "Find records where state is NY",
        "How many active cases are registered?",
        "Compare count of active vs closed status cases"
    ]
    # Filter fallbacks just in case
    fallback_textual = [f for f in fallback_textual if not any(re.search(rf"\b{kw}\b", f.lower()) for kw in visual_kws)]
    
    fallback_visual = [
        "Pie chart of chapter breakdown",
        "Bar chart of cases by state",
        "Line chart of filings over time",
        "Horizontal bar chart of top 10 attorneys",
        "Donut chart of case status",
        "Histogram of match scores"
    ]
    
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
            recent = history[-3:]
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


        prompt = (
            "You are an expert AI analyst for a Bankruptcy Analytics Dashboard.\n\n"

            "OBJECTIVE:\n"
            "Generate intelligent, business-focused follow-up questions and visualization suggestions "
            "based on:\n"
            "1. Database schema\n"
            "2. Current query result dataset\n"
            "3. Conversation history\n"
            "4. Previously explored insights\n\n"

            f"DATABASE SCHEMA:\n{json.dumps(schema, indent=2)[:2000]}\n\n"
            f"{history_str}\n"
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

            "QUESTION GENERATION RULES:\n"

            "A. TEXTUAL QUESTIONS\n"
            "- Generate EXACTLY 4 questions.\n"
            "- Maximum 12 words each.\n"
            "- MUST be direct follow-up questions based on the current query/last user question and current query results.\n"
            "- Business-focused and insight-driven.\n"
            "- Questions must request data, metrics, comparisons, rankings, trends, summaries, "
            "aggregations, filters, or anomalies that follow up logically on the previous query.\n"
            "- strictly Avoid simple questions like \"What is the total number of bankruptcy cases in the database?\".\n"
            "- MUST be answerable using the available schema and current dataset.\n"
            "- MUST reference actual columns, categories, or values from the last query results whenever possible (e.g. if the last query filtered or grouped by certain states, years, chapters, or statuses, ask about specific states, years, chapters, or statuses from that result).\n"
            "- Avoid repeating previously asked questions.\n"
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

            "B. VISUALIZATION SUGGESTIONS\n"
            "- Generate EXACTLY 4 suggestions.\n"
            "- Maximum 12 words each.\n"
            "- Explicitly mention ONE supported chart type.\n"
            "- Must be business-relevant and actionable.\n"
            "- Must use actual columns, categories, or values from the dataset.\n"
            "- Prioritize trend analysis, distributions, rankings, comparisons, correlations, "
            "and geographic insights.\n"
            "- Use phrases such as:\n"
            "  Bar chart of...\n"
            "  Line chart showing...\n"
            "  Donut chart for...\n"
            "  Heatmap of...\n"
            "  Scatter plot comparing...\n\n"

            "CONTEXT AWARENESS:\n"
            "- If this is the first interaction, suggest exploratory questions and charts.\n"
            "- If prior results exist, generate deeper investigative follow-ups.\n"
            "- Use actual values from the latest result summary whenever available.\n"
            "- Focus on uncovering patterns, concentration risks, regional trends, filing behavior, "
            "industry impacts, attorney performance, chapter distribution, court activity, "
            "and temporal changes.\n\n"

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
        logger.info("Generated suggestions via Haiku: %s", response)

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
            recent = conversation_memory["history"][-3:]
            lines = []
            for entry in recent:
                lines.append(f"  User: {entry.get('user_question', '')}")
                if entry.get("sql_query"):
                    lines.append(f"  SQL result: {entry.get('record_count', 0)} rows")
            conv_context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            "You are a smart query understanding assistant for a bankruptcy case management dashboard.\n"
            "Your job is to analyze the user's question and map it to the database schema below.\n\n"
            f"DATABASE SCHEMA:\n{schema_text}\n\n"
            f"{conv_context}"
            "USER QUESTION: \"" + user_query + "\"\n\n"
            "KEY COLUMN MAPPING RULES (use these to resolve ambiguity):\n"
            "- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' → use the 'status' column (values: Active, Closed, Dismissed, Converted, Pending, Discharged). Note: if the query specifically requests a comparison/breakdown between specific statuses (like 'active vs closed'), only return those specific statuses.\n"
            "- 'active/inactive flag' → use the 'active_status' column\n"
            "- 'new/closed/reopened/update record stage' → use the 'record_type' column (values: New, Closed, Reopened, Update)\n"
            "- 'bankruptcy chapter', 'chapter 7/11/13' → use the 'chapter' column\n"
            "- 'filing date', 'when filed' → use the 'date_filed' column\n"
            "- 'debtor name' → use 'first_name' and 'last_name'\n"
            "- 'state' (debtor location) → use the 'state' column\n"
            "- 'attorney', 'lawyer' → use the 'Attorny_First_Name', 'Attorny_lastt_Name' or 'Attorny_DBA' columns. When analyzing or grouping by attorney, ask to show 'Attorny_DBA' (or include first and last names along with 'Attorny_DBA') to represent different practitioners/firms.\n"
            "- 'year', 'yearly', 'by year', 'annual' → use the 'date_filed' or 'open_date' column (or date columns) and extract the year using SQLite's strftime('%Y', ...) function.\n\n"
            "INSTRUCTIONS:\n"
            "1. Understand the user's intent (data_retrieval, aggregation, filter, visualization, follow_up, or unclear).\n"
            "2. Rewrite the question as a clear, unambiguous PLAIN ENGLISH query using EXACT column names from the schema.\n"
            "   IMPORTANT: normalized_query must be a natural language sentence, NOT SQL code.\n"
            "   - Map informal terms to schema columns using the KEY COLUMN MAPPING RULES above.\n"
            "   - Use ONLY column names that exist in the schema. Never invent column names.\n"
            "   - Expand abbreviations and correct spelling based on schema knowledge.\n"
            "   - Preserve all filters, conditions, aggregation requests, AND visualization keywords.\n"
            "   - If the user asked for a 'bar chart', 'pie chart', 'plot', etc., keep those words in the normalized_query.\n"
            "   - Example: 'bar chart of filings by state' → 'Show a bar chart of the count grouped by state column'\n"
            "   - Example: 'show debtors in NY' → 'Retrieve records where state = NY showing first_name, last_name, city, state'\n"
            "   - Example: 'active vs closed case breakdown' → 'Show count of cases grouped by status column filtered to show only status is Active or Closed'\n"
            "   - Example: 'Analyze filings by year and status' → 'Retrieve counts of records grouped by the year (extracted using strftime from date_filed or open_date column) and status column'\n"
            "   - Example: 'Show chapter distribution for each state' → 'Retrieve counts of records grouped by state and chapter columns showing both columns'\n"
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


def generate_sql_from_question(user_question, schema, conversation_memory=None, token_usage=None, main_schema=None):
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

        # Build a human-readable column listing with descriptions for the prompt
        def _format_columns_for_prompt(columns):
            lines = []
            for col in columns:
                col_name = col.get('name') or col.get('Column Name', '')
                col_type = col.get('type') or col.get('Data Type', 'string')
                col_desc = col.get('description') or col.get('Description', '')
                lines.append(f"  - {col_name} ({col_type}): {col_desc}")
            return "\n".join(lines)

        prompt = (
            "You are a Senior SQL Analyst specializing in bankruptcy analytics.\n"
            "Your task is to translate natural language user questions into a single SQLite-compatible SQL query.\n"
            "All dates in the database are stored in YYYY-MM-DD format.\n"
            "Return only valid JSON with a single key `sql` whose value is the SQL string.\n"
            "To filter date columns (such as open_date, date_filed, status_date, etc.) by year, use SQLite's strftime function, e.g. strftime('%Y', open_date) = '2024'.\n"
            "Do NOT include explanations, comments, or additional fields. Ensure the SQL is fully compatible with SQLite.\n\n"
            "CRITICAL COLUMN RULES:\n"
            "1. You MUST ONLY use column names that are EXACTLY listed in the schema below. Do NOT invent, guess, or fabricate column names.\n"
            "2. Column names are CASE-SENSITIVE in the schema. Use the EXACT casing shown (e.g., 'status' not 'Status', 'active_status' not 'Active_Status').\n"
            "3. READ the column descriptions carefully to understand what each column represents before choosing which column to use.\n"
            "4. SEMANTIC MAPPING RULES (use these to pick the correct column):\n"
            "   - 'active vs closed', 'case status', 'open/closed/dismissed/discharged' → use the 'status' column\n"
            "   - 'active/inactive flag' → use the 'active_status' column\n"
            "   - 'new/closed/reopened/update record stage' → use the 'record_type' column\n"
            "   - 'bankruptcy chapter', 'chapter 7/11/13' → use the 'chapter' column\n"
            "   - 'filing date', 'when filed' → use the 'date_filed' column\n"
            "   - 'debtor name' → use 'first_name' and 'last_name' (or pd_ prefixed variants)\n"
            "   - 'state' (for debtor location) → use the 'state' column\n"
            "   - 'attorney', 'lawyer' → use 'Attorny_First_Name', 'Attorny_lastt_Name', or 'Attorny_DBA'. When grouping, analyzing, or counting filings 'by attorney', group by 'Attorny_DBA' (or combine individual name columns with 'Attorny_DBA') to ensure distinct firms/practitioners are shown and the output is meaningful.\n"
            "5. SPECIFIC FILTER/COMPARISON RULES:\n"
            "   - If the user query specifies a comparison, vs, or breakdown between specific values of a column (e.g., 'active vs closed', 'chapter 7 vs 13', 'New vs Closed'), you MUST filter the query using WHERE/HAVING to include ONLY those specific values. For example, for 'active vs closed case breakdown', filter status using `TRIM(status) IN ('Active', 'Closed')`.\n"
            "   - If the user query does NOT specify specific comparison values for status or other columns, do NOT add filters limiting the column values (e.g. do not filter status to ('Active', 'Closed') unless requested).\n"
            "6. If unsure which column to use, ALWAYS prefer the column whose description best matches the user's intent.\n"
            "7. YEAR GROUPING RULES:\n"
            "   - If the user query requests analysis, breakdown, or grouping 'by year' or 'yearly' (e.g. 'filings by year', 'trends by year'), you MUST extract the year from date columns (preferring 'date_filed' or 'open_date' columns depending on availability and description) using SQLite's strftime('%Y', date_column_name) or substr(date_column_name, 1, 4), name it `year`, include it in the SELECT list, and group by it (e.g. `SELECT strftime('%Y', date_filed) as year, status, count(*) ... GROUP BY year, status`).\n"
            "8. COLUMN DISTRIBUTION/BREAKDOWN RULES (CRITICAL):\n"
            "   - If the user query asks for a distribution, breakdown, ratio, or comparison of a column (such as chapter, status, record_type, etc.) 'for each' or 'by' another column (such as state, client, year, etc.) (e.g. 'chapter distribution for each state', 'case status breakdown by client'), you MUST select BOTH columns (the distributed column like 'chapter' and the grouping column like 'state') in the SELECT clause, group by BOTH columns in the GROUP BY clause, and count the total cases (e.g., `SELECT state, chapter, count(*) AS count FROM uploaded_data GROUP BY state, chapter`). NEVER select or group by only one of them when a breakdown/distribution of X by Y is requested. Failure to select and group by both columns is a critical violation.\n\n"
        )

        if main_schema and schema and schema.get('table_name') != main_schema.get('table_name'):
            prompt += (
                f"You have access to two tables:\n"
                f"1. The MAIN table: `{main_schema['table_name']}`\n"
                f"   Main table columns:\n{_format_columns_for_prompt(main_schema['columns'])}\n\n"
                f"2. The TEMPORARY table `{schema['table_name']}` containing the results of the previous query.\n"
                f"   Temporary table columns:\n{_format_columns_for_prompt(schema['columns'])}\n\n"
                f"CRITICAL GUIDELINE:\n"
                f"- If the user's follow-up request refers to or queries columns, aggregates, or details that are ONLY in the main table, "
                f"you should query the main table `{main_schema['table_name']}` using filters matching the context of previous queries in the history.\n"
                f"- If the user's request filters, sorts, plots, or aggregates the current results shown (which are in the temporary table), "
                f"you should query the temporary table `{schema['table_name']}`.\n\n"
            )
        else:
            prompt += (
                f"Database table: {schema['table_name']}\n"
                f"Available columns (ONLY use these exact names):\n{_format_columns_for_prompt(schema['columns'])}\n\n"
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
   - 'record stage' (New/Closed/Reopened/Update) should use 'record_type'
   - 'active/inactive flag' should use 'active_status'
4. Check if the query limits the results to only the specified comparison categories if the user asked for a breakdown/comparison between specific categories (e.g., if the user asked for 'active vs closed breakdown', the SQL must filter status to 'Active' and 'Closed' using a WHERE/HAVING clause like TRIM(status) IN ('Active', 'Closed')). If the user did NOT request filtering by specific categories, the SQL should NOT include such filters.
5. Check if the user's question asks for analysis, counts, or grouping 'by year' or 'yearly'. If so, the query must extract the year using strftime('%Y', date_column) or substr(date_column, 1, 4), name it `year`, include it in the SELECT list, and group by it. If the query fails to group by year when requested, it is INVALID.
6. Check if the user's question asks for a distribution, breakdown, ratio, or comparison of one column (e.g. chapter) by/for/each another column (e.g. state) (e.g., 'chapter distribution for each state'). If so, the query MUST select and group by BOTH columns (e.g. selecting both state and chapter, grouping by both state and chapter), not just one. If the query only selects or groups by one column, it is INVALID and you MUST repair it to select both columns in the SELECT clause and group by both in the GROUP BY clause.
7. Check if the date formats are correct (YYYY-MM-DD) if dates are involved.
{rule_5}
9. Do not include ';' at the end of the query
10. Reject any queries that attempt to modify data

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT * FROM ... WHERE ..."}}

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
    n = n.replace("attorny", "attorney")
    n = n.replace("lastt", "last")
    n = n.replace("addl_1", "address_line_1")
    n = n.replace("addl_2", "address_line_2")
    n = n.replace("ac_no", "account_number")
    n = n.replace("client", "client_name")
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
    """Detect if the user is asking a follow-up question.

    A query is a follow-up ONLY if it refers back to the active result set via
    pronouns / reference words, OR is a pure chart request with no new
    entity/column/filter that would require a fresh main-DB query.
    """
    if not st.session_state.temp_table_name:
        return False

    query_lower = user_query.lower()

    # --- Standard reference-word follow-ups ---
    followup_keywords = [
        'that', 'those', 'these', 'records', 'results', 'rows', 'items',
        'from that', 'from there', 'among those', 'above', 'convert', 'previous',
    ]
    if any(kw in query_lower for kw in followup_keywords):
        return True

    # --- Chart request: only treat as follow-up if it is a *pure* re-chart ---
    # i.e. "plot this", "pie chart", "bar chart" WITHOUT naming a new entity
    # (year, state, attorney, chapter, district, debtor, filing, etc.) that
    # would require going back to the main database.
    if is_chart_request(user_query):
        NEW_ENTITY_HINTS = [
            'year', 'month', 'quarter', 'state', 'district', 'chapter',
            'attorney', 'debtor', 'filing', 'case', 'judge', 'trustee',
            'top', 'bottom', 'by state', 'by year', 'by chapter', 'by district',
            'by attorney', 'over time', 'trend',
        ]
        if any(hint in query_lower for hint in NEW_ENTITY_HINTS):
            # Needs a fresh query against the main DB — NOT a follow-up
            return False
        # No new entity hints → pure re-chart of the active result set
        return True

    return False



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
            clarification_msg = f"❓ {understanding['clarification_needed']}"
            st.warning(clarification_msg)
            _append_conversation_memory(user_query, "", None, answer=clarification_msg)
            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
            return

        # Use the normalized query for all downstream processing
        effective_query = understanding["normalized_query"] if understanding["normalized_query"] else user_query
        # ── End Smart Query Understanding ──────────────────────────────────────

        # NOTE: keyword-based detection always uses original user_query
        # effective_query (schema-normalized) is used ONLY for SQL generation + validation
        is_followup = is_followup_question(user_query, st.session_state.conversation_memory) or (understanding.get("intent") == "follow_up")
        
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

        cache_key = _make_query_cache_key(effective_query, working_schema, st.session_state.conversation_memory)

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
                sql_query = generate_sql_from_question(
                    effective_query,
                    working_schema,
                    st.session_state.conversation_memory,
                    token_usage=generation_token_usage,
                    main_schema=main_db_schema,
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
                validation_result = validate_sql_with_judge(
                    sql_query, effective_query, working_schema, main_schema=main_db_schema
                )
            
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
                st.success(" ")

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
        st.success("")
        st.dataframe(format_table(result_df), width='stretch', hide_index=True)
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
                st.info(f" Result set stored as temporary table. You can ask follow-up questions about these {len(result_df)} records.")
        
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
    
    # Initialize suggested question states
    if "suggested_question_selected" not in st.session_state:
        st.session_state.suggested_question_selected = None

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

    # -------------------------------------------------------------------------
    # DYNAMIC AUTO PROMPT SUGGESTIONS (TEXTUAL & VISUAL) via Bedrock Haiku
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
        </style>
        """,
        unsafe_allow_html=True
    )

    col_t, col_v = st.columns([1, 1])

    with col_t:
        st.markdown('<div class="suggest-anchor-textual"></div>', unsafe_allow_html=True)
        st.caption(" Follow-up questions")
        for idx, txt_q in enumerate(suggestions.get("textual", [])):
            if st.button(txt_q, key=f"suggest_text_{idx}", use_container_width=True):
                st.session_state.suggested_question_selected = txt_q
                st.rerun()

    with col_v:
        st.markdown('<div class="suggest-anchor-visual"></div>', unsafe_allow_html=True)
        st.caption(" Chart suggestions")
        for idx, vis_q in enumerate(suggestions.get("visual", [])):
            if st.button(vis_q, key=f"suggest_visual_{idx}", use_container_width=True):
                st.session_state.suggested_question_selected = vis_q
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
                st.rerun()

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

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]
        query_text = last_msg["content"]

        with st.chat_message("assistant", avatar="blank.png"):
            with st.spinner("Processing request..."):
                _handle_user_query(query_text, schema, active_schema)
            st.rerun()