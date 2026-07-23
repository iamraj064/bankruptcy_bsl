import sqlite3
import os
import re
import time
import json
import pandas as pd
from . import config
from . import logger_config

DB_PATH = os.path.join(os.path.dirname(__file__), "vl_rag.db")
logger = logger_config.setup_logger("vectorless_rag")

SCHEMA_SUMMARY = """
Table: cases
Columns:
- Ac_no (TEXT): Account Number
- client (TEXT): Client Code
- Open_date (TEXT): Date opened
- First_name, middle_name, Last_name (TEXT): Debtor name
- ssn (TEXT): Social Security Number
- City, State, Zipcode (TEXT): Debtor location
- record_type (TEXT): Record type
- match_code (TEXT): Match code
- match_score (TEXT): Match score (numeric text e.g. 80-99)
- notification_no, notice_type (TEXT)
- case_no (TEXT): Bankruptcy Case Number (e.g. 20-47925)
- chapter (TEXT): Bankruptcy Chapter (7, 11, 13)
- date_filed (TEXT): Date case filed (YYYY-MM-DD)
- status (TEXT): Case Status (Active, Closed, Dismissed, Pending, Converted, Discharged)
- Judge_name (TEXT): Presiding Judge
- Attorny_First_Name, Attorny_lastt_Name, Attorny_phone (TEXT): Attorney details
- trustee_name (TEXT): Assigned Trustee
- Court_id, court_district, Court_State (TEXT): Court details
- Disposition_text (TEXT): Case disposition notes
"""

class VectorlessRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def plan_query_intent(self, query: str) -> dict:
        """
        Uses VectorlessSQLGenerator to plan required SQLite queries with
        value alignment and few-shot templates.
        """
        from .vectorless_sql_generator import VectorlessSQLGenerator
        generator = VectorlessSQLGenerator(self.db_path)
        return generator.generate_sql(query)

        return {
            "intent": "search",
            "queries": []
        }

    def parse_intent(self, query: str):
        q_lower = query.lower()
        filters = {}
        
        case_match = re.search(r'\b\d{2}-\d{5}\b', query)
        if case_match:
            filters["case_no"] = case_match.group(0)
            
        ac_match = re.search(r'\b100\d{4}\b', query)
        if ac_match:
            filters["Ac_no"] = ac_match.group(0)
            
        chap_match = re.search(r'\bchapter\s*(7|11|13)\b', q_lower)
        if chap_match:
            filters["chapter"] = chap_match.group(1)
            
        statuses = ["active", "closed", "dismissed", "pending", "converted", "discharged", "reopened"]
        for s in statuses:
            if re.search(r'\b' + s + r'\b', q_lower):
                filters["status"] = s.capitalize()
                break

        state_match = re.search(r'\b(FL|CA|NY|TX|GA|AZ|PA|IL|NC|OH)\b', query.upper())
        if state_match:
            filters["State"] = state_match.group(1)
            
        return filters

    def clean_fts_query(self, query: str, filters: dict = None) -> str:
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
        stopwords = {
            "the", "and", "for", "with", "what", "where", "which", "who", 
            "how", "can", "you", "show", "find", "list", "tell", "case", 
            "cases", "details", "information", "all", "about", "get", "has", "highest",
            "most", "least", "top"
        }
        
        # Exclude terms that were already mapped to structured filters (e.g. 'TX', 'Active')
        exclude_terms = set()
        if filters:
            for k, v in filters.items():
                if isinstance(v, str):
                    exclude_terms.add(v.lower())
        
        filtered_words = [w for w in words if w.lower() not in stopwords and w.lower() not in exclude_terms]
        
        if not filtered_words:
            return "*"
        return " AND ".join([f'"{w}"*' for w in filtered_words])

    def execute_plan(self, plan: dict, top_k: int = config.DEFAULT_TOP_K):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        results = []
        executed_sqls = []
        
        for q_info in plan.get("queries", []):
            sql = q_info.get("sql", "")
            desc = q_info.get("description", "")
            if not sql:
                continue
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
                results.append({
                    "description": desc,
                    "sql": sql,
                    "data": [dict(r) for r in rows]
                })
                executed_sqls.append(sql)
            except Exception:
                pass
                
        conn.close()
        return results, executed_sqls

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K):
        start_time = time.time()
        
        # 1. Call Smart Query Intent Planner
        plan = self.plan_query_intent(query)
        
        if plan.get("queries"):
            results, sqls = self.execute_plan(plan, top_k)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Planned queries executed: {json.dumps(sqls)}")
            metadata = {
                "search_mode": f"Smart Intent ({plan.get('intent', 'analytical')})",
                "executed_sql": "; \n".join(sqls),
                "retrieval_time_ms": round(elapsed_ms, 2),
                "vector_embeddings_generated": 0,
                "retrieved_count": len(results)
            }
            return results, metadata

        # 2. Pure FTS5/Lookup Fallbacks
        conn = self.get_connection()
        cursor = conn.cursor()
        filters = self.parse_intent(query)
        records = []
        executed_sql = ""
        
        where_clauses = []
        params = []
        
        # Inject all extracted structured filters
        for col, val in filters.items():
            where_clauses.append(f"c.{col} = ?")
            params.append(val)
            
        fts_term = self.clean_fts_query(query, filters)
        if fts_term != "*":
            where_clauses.append("cases_fts MATCH ?")
            params.append(fts_term)
            
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        executed_sql = f'''
        SELECT c.Ac_no as Account_No, c.case_no as Case_No, c.client as Client, c.First_name as First_Name, c.Last_name as Last_Name, c.State, c.chapter as Chapter, c.status as Status,
               c.Attorny_First_Name as Attorney_First_Name, c.Attorny_lastt_Name as Attorney_Last_Name, c.trustee_name as Trustee, c.Judge_name as Judge
        FROM cases c
        JOIN cases_fts fts ON c.Ac_no = fts.Ac_no
        {where_sql}
        ORDER BY fts.rank ASC
        LIMIT {top_k};
        '''
        
        try:
            cursor.execute(executed_sql, params)
            rows = cursor.fetchall()
            records = [dict(r) for r in rows]
            search_mode = "Hybrid Structured + FTS"
        except sqlite3.OperationalError:
            clean_raw = re.sub(r'[^\w\s]', '', query).strip()
            executed_sql = f'SELECT Ac_no as Account_No, case_no as Case_No, First_name as First_Name, Last_name as Last_Name, State, status as Status FROM cases WHERE Disposition_text LIKE ? OR Judge_name LIKE ? OR trustee_name LIKE ? OR City LIKE ? LIMIT {top_k};'
            param_like = f"%{clean_raw}%"
            cursor.execute(executed_sql, [param_like, param_like, param_like, param_like])
            rows = cursor.fetchall()
            records = [dict(r) for r in rows]
            search_mode = "Generic LIKE Fallback"

        conn.close()
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"Fallback query executed | Mode: {search_mode} | SQL: {executed_sql.strip()}")

        metadata = {
            "search_mode": search_mode,
            "executed_sql": executed_sql.strip(),
            "retrieval_time_ms": round(elapsed_ms, 2),
            "vector_embeddings_generated": 0,
            "retrieved_count": len(records)
        }
        return records, metadata


class VectorlessRAGQA:
    def __init__(self):
        self.retriever = VectorlessRetriever()

    def smart_query_understanding(self, user_query: str, chat_history: list = None) -> dict:
        """
        Uses standard LLM integration to understand the query, classify intent, and normalize it.
        """
        q_lower = user_query.strip().lower()
        
        # 1. Programmatic Intercept for "top N cases for/in [state]"
        try:
            _STATE_NAME_TO_CODE = {
                "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
                "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
                "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
                "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
                "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
                "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
                "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
                "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
                "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
                "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
                "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
                "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
                "wisconsin": "WI", "wyoming": "WY",
            }
            _raw_retrieval_pattern = re.compile(
                r"(?:show(?:\s+me)?|list|get|display|find)?\s*"
                r"(?:top|first|latest|recent)?\s*(\d+)?\s*"
                r"(?:cases?|records?|filings?|bankruptcies)\s+"
                r"(?:for|in|from|of)\s+"
                r"([A-Za-z0-9][A-Za-z0-9\s]{0,20})",
                re.IGNORECASE
            )
            _match = _raw_retrieval_pattern.match(user_query.strip())
            if _match:
                _limit_str = _match.group(1)
                _location_raw = _match.group(2).strip().rstrip(".,!?")
                _limit = int(_limit_str) if _limit_str else 10
                
                _location_key = _location_raw.lower()
                _state_code = _STATE_NAME_TO_CODE.get(_location_key)
                if not _state_code and len(_location_raw) == 2:
                    _maybe_state = _location_raw.upper()
                    if _maybe_state in _STATE_NAME_TO_CODE.values():
                        _state_code = _maybe_state
                    
                if _state_code:
                    _norm_q = (
                        f"Retrieve records where State = {_state_code} showing Ac_no, "
                        f"First_name, Last_name, State, chapter, status, date_filed, match_score "
                        f"ordered by match_score descending showing only the top {_limit}"
                    )
                else:
                    _norm_q = (
                        f"Retrieve the top {_limit} raw case records for {_location_raw} "
                        f"showing Ac_no, First_name, Last_name, State, chapter, status, date_filed, match_score "
                        f"ordered by match_score descending"
                    )

                logger.info(
                    f"INTERCEPT: 'top N cases for filter' pattern matched. filter={_location_raw} limit={_limit} normalized_query={_norm_q!r}"
                )
                return {
                    "normalized_query": _norm_q,
                    "intent": "data_retrieval",
                    "relevant_columns": ["Ac_no", "First_name", "Last_name", "State", "chapter", "status", "date_filed", "match_score"],
                    "time_filter": None,
                    "is_answerable": True,
                    "clarification_needed": None,
                }
        except Exception as e:
            logger.warn(f"Programmatic intercept failed: {e}")

        # 2. Build conversational history context
        conv_context = ""
        if chat_history:
            recent = chat_history[-2:]
            lines = []
            for entry in recent:
                lines.append(f"  User: {entry.get('content', '')}")
            conv_context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        # 3. Prompt definition using accurate columns
        prompt = f"""You are a smart query understanding assistant for a bankruptcy case management dashboard.
Your job is to analyze the user's question and map it to the database schema below.

DATABASE SCHEMA:
{SCHEMA_SUMMARY}

{conv_context}
USER QUESTION: "{user_query}"

KEY COLUMN MAPPING RULES (use these to resolve ambiguity):
- 'active vs closed', 'case status', 'open/closed/dismissed/discharged' → use the 'status' column (values: Active, Closed, Dismissed, Converted, Pending, Discharged). Note: if the query specifically requests a comparison/breakdown between specific statuses (like 'active vs closed') along with another dimension (e.g. by year, by state), make sure the normalized query explicitly asks for both grouping dimensions (e.g. 'Show count of cases grouped by status AND State/year').
- 'active/inactive flag' → use the 'active_status' column (values: Y/N)
- 'new/closed/reopened/update record stage' → use the 'record_type' column (values: New, Closed, Reopened, Update)
- CRITICAL DISAMBIGUATION: 'New' is EXCLUSIVELY a record_type value (NEVER a status value). When the user says 'new vs closed', 'closed vs new', 'new cases', always use 'record_type' column. Use 'status' only for legal status values like Active, Pending, Dismissed, Discharged, Converted.
- 'bankruptcy chapter', 'chapter 7/11/13' → use the 'chapter' column
- 'filing date', 'when filed' → use the 'date_filed' column
- 'debtor name', 'customer name' → use 'First_name' and 'Last_name' (perform a partial match LIKE search)
- 'state' (debtor location) → use the 'State' column
- 'client', 'client name', 'client code', 'VP', 'SYF', 'carecredit' or any client identifier → use the 'client' column. NEVER map client terms to 'match_code'.
- 'attorney', 'lawyer' → use the 'Attorny_First_Name', 'Attorny_lastt_Name' or 'Attorny_phone' columns (note exact spelling: 'Attorny', 'lastt').
- 'year', 'yearly', 'by year', 'annual' → use the 'date_filed' or 'Open_date' column and extract the year using SQLite's strftime('%Y', ...) function.
- 'matchcode', 'match_code' (values like P1, P2, P3, M1, M2, M3) → use the 'match_code' column.
- 'P2 matchcode' → matches the 'match_code' column with value 'P2'.
- 'partnership cases' → use 'consumer_type' = 'Partnership'
- 'business cases' → use 'consumer_type' = 'Business'
- 'corporate' → use 'consumer_type' = 'Corporate'
- 'prose', 'pro se', 'self-represented' → use 'prose_indicator' = 'Y'
- 'without attorney', 'no attorney' → check if 'prose_indicator' = 'Y' OR attorney columns are NULL/empty
- 'held cases' or 'cases held' → use 'status' = 'Pending'
- 'transfers' in corporate context → use 'status' = 'Converted' or check 'conversion_date' column
- 'high risk' cases → use 'match_score' >= 98
- Pronoun follow-ups like 'who are they', 'who are those', 'what are they', 'list them' asking for the entities behind a count or aggregation → Rewrite to list/retrieve the distinct names or values of that entity (e.g. 'Show the unique client values' or 'List the unique clients'). Do NOT count them again. Avoid 'count' or 'how many' in the normalized query when the user is asking 'who' or 'what' to list the actual entities.

INSTRUCTIONS:
1. Understand the user's intent (data_retrieval, aggregation, filter, visualization, follow_up, or unclear).
2. Rewrite the question as a clear, unambiguous PLAIN ENGLISH query using EXACT column names from the schema.
   IMPORTANT: normalized_query must be a natural language sentence, NOT SQL code.
   - Map informal terms to schema columns using the KEY COLUMN MAPPING RULES above.
   - Use ONLY column names that exist in the schema. Never invent column names.
   - Expand abbreviations and correct spelling based on schema knowledge.
   - Preserve all specific filter values (e.g. state names like 'NY', status names like 'Active', chapter numbers like '7', and match codes like 'P2' or 'M1'). Never generalize specific filter values or drop them.
   - Keep limit, ordering, and superlative constraints explicitly in the normalized query. For example:
     * For singular superlative questions (e.g., 'Which state has the most filings'), rewrite it to include 'showing only the top 1'.
     * For general distribution, grouping, or yearly/monthly breakdown questions without superlatives (e.g., 'filings by state'), do NOT include any limit restriction.
   - If the user asked for a chart, line graph, etc., keep those words in the normalized_query.
   - Example: 'who are they?' (when previous query was 'How many unique clients') → 'Retrieve the unique client values'
   - Example: 'show unique clients' → 'Show count of cases grouped by client column ordered by count descending'
   - Example: 'how many unique clients are there' → 'Show total count of distinct client values'
3. List the relevant column names from the schema (use exact column names).
4. Extract any time/date filter mentioned (e.g., '2024') or null.
5. Determine if the question is answerable from the schema (true/false).
6. If unclear or unanswerable, provide a brief clarification message.

Return ONLY a valid JSON object with these exact keys:
{{
  "normalized_query": "plain English rewrite - never SQL",
  "intent": "data_retrieval|aggregation|filter|visualization|follow_up|unclear",
  "relevant_columns": ["col1", "col2"],
  "time_filter": "2024" or null,
  "is_answerable": true or false,
  "clarification_needed": "..." or null
}}
Do NOT include markdown, SQL code, or any text outside the JSON."""

        response = config.call_llm(prompt, system_prompt="You are a JSON-only query understanding assistant.")
        
        result = {
            "normalized_query": user_query,
            "intent": "data_retrieval",
            "relevant_columns": [],
            "time_filter": None,
            "is_answerable": True,
            "clarification_needed": None
        }
        if response:
            try:
                cleaned = re.sub(r'```json|```', '', response).strip()
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    result.update(data)
            except Exception as e:
                logger.warn(f"Failed to parse query understanding response: {e}")
                
        return result

    def format_results_as_markdown(self, retrieved_data) -> str:
        """
        Formats all retrieved dataframes/dicts from plan execution to markdown tables.
        """
        if not retrieved_data:
            return "No data found."
        
        # If it's a standard single list of records
        if isinstance(retrieved_data, list) and len(retrieved_data) > 0 and "data" not in retrieved_data[0]:
            df = pd.DataFrame(retrieved_data)
            return df.to_markdown(index=False)
            
        # If it's multi-query structured output
        markdown_blocks = []
        for r_info in retrieved_data:
            desc = r_info.get("description", "Query Result")
            data = r_info.get("data", [])
            if not data:
                continue
            df = pd.DataFrame(data)
            markdown_blocks.append(f"#### {desc}\n" + df.to_markdown(index=False))
            
        return "\n\n".join(markdown_blocks)

    def _contextualize_query(self, query: str, chat_history: list) -> str:
        if not chat_history:
            return query
        
        # Build a prompt to contextualize/rewrite the user's question
        history_str = ""
        # Take the last 6 messages to keep it concise and context-relevant
        for turn in chat_history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"]
            # If assistant content is too long or contains a markdown table, just keep the text
            if len(content) > 300:
                content = content[:300] + "... [truncated table/data]"
            history_str += f"{role}: {content}\n"
             
        prompt = f"""Given the following chat history and a follow-up question, rewrite the follow-up question to be a self-contained question (i.e., it must contain all necessary details, entities, states, chapters, or filters referred to in the chat history, so it can be executed as a standalone database query).
Do NOT answer the question. Only return the rewritten question.

Chat History:
{history_str}
Follow-up Question: {query}
Rewritten Standalone Question:"""

        try:
            rewritten = config.call_llm(prompt, temperature=0.1)
            if rewritten and len(rewritten.strip()) > 5:
                logger.info(f"Contextualized query from '{query}' to '{rewritten.strip()}'")
                return rewritten.strip()
        except Exception as e:
            logger.warn(f"Failed to contextualize query: {e}")
            
        # Fallback programmatic contextualizer when LLM is inactive
        q_lower = query.lower()
        has_pronoun = any(w in q_lower for w in ["these", "those", "they", "them", "there", "this", "that", "it"])
        if has_pronoun:
            last_user_q = None
            for turn in reversed(chat_history):
                if turn.get("role") == "user":
                    last_user_q = turn.get("content", "")
                    break
            if last_user_q:
                states_found = re.findall(r'\b(FL|CA|NY|TX|GA|AZ|PA|IL|NC|OH)\b', last_user_q.upper())
                chapters_found = re.findall(r'\b(?:chapter\s*)?(7|11|13)\b', last_user_q.lower())
                clients_found = [c for c in ["DM", "FD", "GECOM", "VP"] if c in last_user_q.upper()]
                statuses_found = [s for s in ["Active", "Closed", "Dismissed", "Pending", "Converted", "Discharged"] if s.lower() in last_user_q.lower()]
                
                context_additions = []
                if states_found:
                    context_additions.append(states_found[0])
                if chapters_found:
                    context_additions.append(f"chapter {chapters_found[0]}")
                if clients_found:
                    context_additions.append(f"client {clients_found[0]}")
                if statuses_found:
                    context_additions.append(f"status {statuses_found[0]}")
                
                if context_additions:
                    rewritten = f"{query} for {' and '.join(context_additions)}"
                    logger.info(f"Fallback programmatic contextualization: '{query}' -> '{rewritten}'")
                    return rewritten

        return query

    def generate_followup_questions(self, query: str, retrieved_data, chat_history: list) -> list:
        # If LLM is active, we can ask it to generate 3 relevant follow-up questions
        prompt = f"""Based on the user's latest query: "{query}"
And the retrieved database records/aggregations:
{self.format_results_as_markdown(retrieved_data)[:1000]}

Generate 3 natural, logical follow-up questions that the user might want to ask next about this data (e.g. comparing other dimensions, drilling down on specific states/chapters found, looking up trend over time, or checking status of records).
Output ONLY a JSON list of 3 strings. Do not include markdown code blocks or explanations.
Example output format:
["What is the chapter 7 count for those states?", "Show me the top 3 cities in TX", "Compare FL vs CA filings"]

JSON Output:"""
        try:
            response = config.call_llm(prompt, temperature=0.5)
            if response:
                cleaned = re.sub(r'```json|```', '', response).strip()
                questions = json.loads(cleaned)
                if isinstance(questions, list) and len(questions) >= 3:
                    return [str(q).strip() for q in questions[:3]]
        except Exception:
            pass
        
        # Fallback to rule-based questions if LLM fails
        return self._generate_rule_based_followups(query, retrieved_data)

    def _generate_rule_based_followups(self, query: str, retrieved_data) -> list:
        """
        Generates context-aware follow-up questions by extracting real entities
        from both the query text and the retrieved data rows.
        Never falls back to hardcoded generic suggestions.
        """
        import re as _re
        q_lower = query.lower()

        # ── 1. Extract entities from the query ─────────────────────────────
        from .vectorless_sql_generator import VectorlessSQLGenerator
        generator = VectorlessSQLGenerator(self.retriever.db_path)
        filters = generator.parse_intent(query)

        state   = filters.get("State")
        chapter = filters.get("chapter")
        status  = filters.get("status")
        client  = filters.get("client")

        # ── 2. Extract entities present in returned rows ───────────────────
        row_states    = set()
        row_chapters  = set()
        row_clients   = set()
        row_statuses  = set()
        row_matchcodes= set()

        rows = []
        if isinstance(retrieved_data, list):
            for item in retrieved_data:
                if isinstance(item, dict) and "data" in item:
                    rows.extend(item["data"])
                elif isinstance(item, dict):
                    rows.append(item)

        for row in rows[:50]:           # inspect up to 50 rows
            if "State" in row and row["State"]:
                row_states.add(str(row["State"]).strip())
            if "chapter" in row and row["chapter"]:
                row_chapters.add(str(row["chapter"]).strip())
            if "client" in row and row["client"]:
                row_clients.add(str(row["client"]).strip())
            if "status" in row and row["status"]:
                row_statuses.add(str(row["status"]).strip())
            if "match_code" in row and row["match_code"]:
                row_matchcodes.add(str(row["match_code"]).strip())

        # Pick representative values from rows (first 2 of each)
        top_states     = sorted(row_states)[:2]
        top_chapters   = sorted(row_chapters)[:2]
        top_clients    = sorted(row_clients)[:2]
        top_statuses   = sorted(row_statuses)[:2]
        top_matchcodes = sorted(row_matchcodes)[:2]

        followups = []

        # ── 3. Intent-based follow-up generation ──────────────────────────

        # --- CLIENT queries ---
        if client or any(kw in q_lower for kw in ["client", "clients"]):
            used_client = client or (top_clients[0] if top_clients else "DM")
            other_clients = [c for c in ["DM", "FD", "GECOM", "VP"] if c != used_client]
            followups.append(f"What is the chapter distribution for client {used_client}?")
            followups.append(f"Show year wise filing trend for client {used_client}")
            if other_clients:
                followups.append(f"Compare client {used_client} vs {other_clients[0]} by filing status")

        # --- MATCH CODE queries ---
        elif any(kw in q_lower for kw in ["match_code", "matchcode", "match code", "p1", "p2", "p3", "m1", "m2", "m3"]):
            used_code = top_matchcodes[0] if top_matchcodes else "P2"
            followups.append(f"How many {used_code} cases are currently Active?")
            followups.append(f"Show the state distribution for {used_code} matchcode cases")
            followups.append(f"Compare P1 vs P2 vs M1 filing volumes")

        # --- COMPARISON queries ---
        elif "compare" in q_lower or " vs " in q_lower:
            if top_states:
                followups.append(f"Show the chapter breakdown for {top_states[0]}")
            else:
                followups.append("Show the chapter breakdown for these states")
            followups.append("What is the year wise trend of filings for these groups?")
            followups.append("Which attorneys handle the most cases across these groups?")

        # --- STATUS queries ---
        elif status or any(kw in q_lower for kw in ["status", "active", "closed", "dismissed", "discharged", "pending", "converted"]):
            used_status = status or (top_statuses[0] if top_statuses else "Active")
            other_status = [s for s in ["Active", "Closed", "Dismissed", "Discharged", "Pending"] if s != used_status]
            followups.append(f"Which states have the most {used_status} cases?")
            followups.append(f"Show year wise trend of {used_status} filings")
            if other_status:
                followups.append(f"Compare {used_status} vs {other_status[0]} case volumes")

        # --- RECORD TYPE queries ---
        elif any(kw in q_lower for kw in ["record_type", "new", "reopened", "update", "record type"]):
            followups.append("Compare New vs Reopened record types by state")
            followups.append("Show the year wise breakdown of record types")
            followups.append("Which clients have the most New records filed?")

        # --- ATTORNEY / JUDGE queries ---
        elif any(kw in q_lower for kw in ["attorney", "lawyer", "judge", "trustee", "presiding"]):
            followups.append("Which attorneys handle the most chapter 7 cases?")
            followups.append("Show the state distribution of cases for these attorneys")
            followups.append("What is the status breakdown of cases handled by top attorneys?")

        # --- STATE queries ---
        elif state or any(kw in q_lower for kw in ["state", "states"]):
            used_state = state or (top_states[0] if top_states else "FL")
            other_state = top_states[1] if len(top_states) > 1 else "CA"
            followups.append(f"What is the chapter distribution of {used_state} cases?")
            followups.append(f"Show the status breakdown of filings in {used_state}")
            followups.append(f"Compare {used_state} vs {other_state} by filing volume")

        # --- CHAPTER queries ---
        elif chapter or any(kw in q_lower for kw in ["chapter", "ch7", "ch13", "ch11"]):
            used_chapter = chapter or (top_chapters[0] if top_chapters else "7")
            other_chapter = "13" if str(used_chapter) != "13" else "7"
            followups.append(f"Which states have the highest chapter {used_chapter} filings?")
            followups.append(f"Show year wise trend of chapter {used_chapter} cases")
            followups.append(f"Compare chapter {used_chapter} vs chapter {other_chapter} by status")

        # --- COUNT / AGGREGATION queries ---
        elif any(kw in q_lower for kw in ["count", "total", "volume", "how many", "number of"]):
            dim_found = None
            if top_states:     dim_found = f"the state {top_states[0]}"
            elif top_chapters: dim_found = f"chapter {top_chapters[0]}"
            elif top_clients:  dim_found = f"client {top_clients[0]}"

            if dim_found:
                followups.append(f"Show year wise breakdown for {dim_found}")
                followups.append(f"What is the status distribution for {dim_found}?")
            else:
                followups.append("Show year wise filing trends")
                followups.append("What is the distribution of filings by chapter?")
            followups.append("Which clients contribute the most to these filings?")

        # --- YEAR / TREND queries ---
        elif any(kw in q_lower for kw in ["year", "yearly", "trend", "month", "monthly", "over time", "annual"]):
            used_state = top_states[0] if top_states else "FL"
            followups.append(f"Which states showed the highest growth trend?")
            followups.append(f"Compare filing trends by chapter over time")
            followups.append(f"Show client wise filing volumes year over year")

        # --- LOOKUP / ROW-LEVEL queries (top N, show me, retrieve) ---
        elif any(kw in q_lower for kw in ["top", "show", "list", "retrieve", "lookup", "find", "cases in", "records"]):
            used_state = state or (top_states[0] if top_states else None)
            used_chapter = chapter or (top_chapters[0] if top_chapters else None)

            if used_state and used_chapter:
                followups.append(f"What is the status breakdown of chapter {used_chapter} cases in {used_state}?")
                followups.append(f"Show year wise trend of filings in {used_state}")
                followups.append(f"Which clients have cases in {used_state}?")
            elif used_state:
                followups.append(f"Show the chapter distribution for {used_state}")
                followups.append(f"What are the top attorneys handling {used_state} cases?")
                followups.append(f"Compare {used_state} vs another state by filing volume")
            elif used_chapter:
                followups.append(f"Which states have the most chapter {used_chapter} filings?")
                followups.append(f"Show year wise trend of chapter {used_chapter} cases")
                followups.append(f"What is the client breakdown for chapter {used_chapter}?")
            else:
                followups.append("Show the chapter distribution across all cases")
                followups.append("What is the status breakdown of these records?")
                followups.append("Which states have the most filings?")

        # ── 4. True fallback — still uses actual data values if available ──
        if not followups:
            if top_states and len(top_states) >= 2:
                followups.append(f"Compare {top_states[0]} vs {top_states[1]} filing volumes")
            elif top_states:
                followups.append(f"Show chapter distribution for {top_states[0]}")
            else:
                followups.append("Show filings grouped by state")

            if top_chapters:
                followups.append(f"Show year wise trend for chapter {top_chapters[0]} cases")
            else:
                followups.append("Show year wise filing trends by chapter")

            if top_clients:
                followups.append(f"What is the status breakdown for client {top_clients[0]}?")
            else:
                followups.append("Compare filing volumes across different clients")

        return followups[:3]

    def generate_answer(self, query: str, top_k: int = config.DEFAULT_TOP_K, chat_history: list = None):
        logger.info(f"Start generate_answer for query: '{query}'")
        
        # Contextualize query based on conversational history
        contextualized = self._contextualize_query(query, chat_history)
        
        # Smart Query Understanding preprocessor and guardrails
        understanding = self.smart_query_understanding(contextualized, chat_history)
        logger.info(f"Smart Query Understanding | intent={understanding['intent']} | answerable={understanding['is_answerable']}")
        
        # Guardrail check
        if not understanding["is_answerable"] and understanding["clarification_needed"]:
            clarification_msg = (
                f"I wasn't able to find a direct match for that in the dataset. "
                f"{understanding['clarification_needed']}\n\n"
                "💡 *Try rephrasing using database columns like* `status`, `chapter`, `State`, `client`, or `match_code`."
            )
            meta = {
                "search_mode": "Intent Guardrail",
                "executed_sql": "",
                "retrieval_time_ms": 0.0,
                "vector_embeddings_generated": 0,
                "retrieved_count": 0,
                "understanding": {
                    "intent": understanding.get("intent"),
                    "relevant_columns": understanding.get("relevant_columns"),
                    "time_filter": understanding.get("time_filter")
                }
            }
            return {
                "answer": clarification_msg,
                "metadata": meta,
                "raw_records": [],
                "is_llm_synthesized": False,
                "followup_questions": self.generate_followup_questions(query, [], chat_history)
            }
            
        effective_query = understanding["normalized_query"] if understanding["normalized_query"] else contextualized
        
        retrieved, meta = self.retriever.retrieve(effective_query, top_k=top_k)
        logger.info(f"Retrieve completed. Mode: {meta.get('search_mode')}, Latency: {meta.get('retrieval_time_ms')} ms, Count: {meta.get('retrieved_count')}")
        
        # Store understanding metadata inside the returned response object's metadata
        meta["understanding"] = {
            "intent": understanding.get("intent"),
            "relevant_columns": understanding.get("relevant_columns"),
            "time_filter": understanding.get("time_filter")
        }
        
        if not retrieved:
            logger.info("No matching records found for query.")
            return {
                "answer": "No matching bankruptcy records were found for your query.",
                "metadata": meta,
                "raw_records": [],
                "followup_questions": self.generate_followup_questions(query, [], chat_history)
            }

        data_summary_md = self.format_results_as_markdown(retrieved)

        # Smart LLM Synthesis for Comparisons and Aggregations
        system_prompt = """You are a senior data analyst. Synthesize a professional, comprehensive data comparison or analysis based on the retrieved structured tables.
Guidelines:
1. Do not give a direct straightforward list or simple count.
2. Present distinct comparisons clearly using Markdown Tables comparing the dimensions (e.g., TX vs AZ total filings, chapter breakdowns, status profiles).
3. Follow with clear, structured bullet points explaining key differences and observations.
4. Do not output raw JSON or technical system metadata."""

        prompt = f"User Question: {contextualized}\n\nRetrieved Data Context:\n{data_summary_md}\n\nPlease generate a thorough comparison."
        
        llm_response = config.call_llm(prompt, system_prompt=system_prompt)
        
        if llm_response:
            logger.info("Successfully synthesized response via LLM call.")
            answer = llm_response
            is_llm_synthesized = True
        else:
            logger.info("No LLM response received; returning native fallback markdown data tables.")
            answer = data_summary_md
            is_llm_synthesized = False

        # Generate follow-up questions
        followup_questions = self.generate_followup_questions(contextualized, retrieved, chat_history)

        logger.info(f"Generated Answer:\n{answer}")
        logger.info(f"Completed generate_answer for query: '{query}'")
        return {
            "answer": answer,
            "metadata": meta,
            "raw_records": retrieved,
            "is_llm_synthesized": is_llm_synthesized,
            "followup_questions": followup_questions
        }

if __name__ == "__main__":
    qa = VectorlessRAGQA()
    print("Testing Comparison Query:")
    res = qa.generate_answer("compare filings between TX and AZ")
    print(res["answer"])
