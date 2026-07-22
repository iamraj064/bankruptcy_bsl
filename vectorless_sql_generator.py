import sqlite3
import re
import json
import config
import logger_config

DB_PATH = "vl_rag.db"
logger = logger_config.setup_logger("sql_generator")

# Schema catalog for query generation
TABLE_SCHEMA = {
    "table": "cases",
    "columns": {
        "Ac_no": "Account Number (unique identifier, e.g. 1000001)",
        "client": "Client Code (e.g. DM, FD, VP, GECOM)",
        "First_name": "Debtor's First Name",
        "middle_name": "Debtor's Middle Name",
        "Last_name": "Debtor's Last Name",
        "ssn": "Social Security Number (masked/raw)",
        "City": "Debtor's City",
        "State": "Debtor's State code (e.g. FL, CA, NY)",
        "Zipcode": "Debtor's Zipcode",
        "record_type": "Record Type (New, Update, Closed, Reopened)",
        "match_code": "Match Code (P1, P2, P3, M1, M2, M3)",
        "match_score": "Confidence score (integer text 80-99)",
        "case_no": "Bankruptcy Case Number (e.g. 20-47925)",
        "chapter": "Bankruptcy Chapter (7, 11, 13)",
        "date_filed": "Case filing date (YYYY-MM-DD)",
        "status": "Case Status (Active, Closed, Converted, Discharged, Dismissed, Pending)",
        "Judge_name": "Presiding Judge Name",
        "prose_indicator": "Pro Se Debtor Indicator (Y/N)",
        "trustee_name": "Assigned Trustee Name",
        "Court_id": "Court Identifier",
        "court_district": "Court District Description",
        "Court_State": "State where court resides",
        "Disposition_text": "Case disposition text status (Appealed, Case Closed, Case in Progress, Under Review)"
    }
}

# Pre-validated SQL examples to guide the generation (Few-shot repository)
FEW_SHOT_TEMPLATES = [
    {
        "category": "comparison",
        "question": "compare filings between TX and AZ",
        "sql": [
            "SELECT State, COUNT(*) as Total_Filings FROM cases WHERE State IN ('TX', 'AZ') GROUP BY State;",
            "SELECT State, chapter as Chapter, COUNT(*) as Count FROM cases WHERE State IN ('TX', 'AZ') GROUP BY State, Chapter ORDER BY State, Chapter;"
        ]
    },
    {
        "category": "aggregation",
        "question": "which judge handled the most cases in chapter 7",
        "sql": [
            "SELECT Judge_name, COUNT(*) as Case_Count FROM cases WHERE chapter = '7' GROUP BY Judge_name ORDER BY Case_Count DESC LIMIT 5;"
        ]
    },
    {
        "category": "aggregation",
        "question": "Which attorneys handle the most cases across these groups?",
        "sql": [
            "SELECT TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) as Attorney, COUNT(*) as Total_Cases FROM cases WHERE TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) != '' GROUP BY Attorney ORDER BY Total_Cases DESC LIMIT 10;"
        ]
    },
    {
        "category": "lookup",
        "question": "show cases handled by counselor lawyer for debtor janet evans",
        "sql": [
            "SELECT Ac_no, case_no, First_name, Last_name, Attorny_First_Name, Attorny_lastt_Name FROM cases WHERE Attorny_First_Name = 'Counselor' AND Attorny_lastt_Name = 'Lawyer' AND First_name = 'Janet' AND Last_name = 'Evans';"
        ]
    }
]

class VectorlessSQLGenerator:
    """
    Leverages Vectorless RAG techniques (FTS5 Value Matching, Metadata Schema Alignment,
    and Few-shot SQL templates) to generate precise SQLite queries.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def lookup_database_values(self, query: str) -> list:
        """
        Runs an FTS5 search on query tokens to find actual column values
        matching the query entities, preventing column hallucination.
        """
        # Clean query words (allow 2-letter words like TX, AZ)
        words = re.findall(r'\b\w{2,}\b', query.lower())
        stopwords = {
            "show", "find", "list", "cases", "filings", "between", "under", "with", 
            "where", "what", "how", "and", "the", "are", "for", "out", "about", "compare",
            "in", "by", "to", "of", "on", "at", "if", "is", "it", "he", "me", "my", 
            "we", "do", "so", "up", "no", "or", "as", "us", "an"
        }
        search_words = [w for w in words if w not in stopwords]
        
        if not search_words:
            return []
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        matches = []
        # Search key columns for value exact match alignment
        # States, Chapters, Cities, Trustee Names, Judges, Clients, Attorney names
        for word in search_words:
            fts_query = f'"{word}"*'
            sql = """
            SELECT c.*
            FROM cases c
            JOIN cases_fts fts ON c.Ac_no = fts.Ac_no
            WHERE fts.cases_fts MATCH ?
            LIMIT 3;
            """
            try:
                cursor.execute(sql, [fts_query])
                rows = cursor.fetchall()
                for row in rows:
                    row_dict = dict(row)
                    for col, val in row_dict.items():
                        if val and word in str(val).lower():
                            matches.append({
                                "search_term": word,
                                "matched_column": col,
                                "matched_value": val
                            })
            except Exception as e:
                # Log or print error for debugging
                pass
                
        conn.close()
        
        # Deduplicate matches
        unique_matches = []
        seen = set()
        for m in matches:
            key = (m["matched_column"], m["matched_value"])
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)
                
        return unique_matches[:6]

    def get_few_shot_examples(self, query: str) -> list:
        """
        Retrieves relevant SQL query examples matching query intent.
        """
        q_lower = query.lower()
        examples = []
        
        if "compare" in q_lower or "versus" in q_lower or " vs " in q_lower:
            # Match comparison category
            examples = [ex for ex in FEW_SHOT_TEMPLATES if ex["category"] == "comparison"]
        elif any(kw in q_lower for kw in ["most", "highest", "count", "average", "total"]):
            examples = [ex for ex in FEW_SHOT_TEMPLATES if ex["category"] == "aggregation"]
        else:
            examples = [ex for ex in FEW_SHOT_TEMPLATES if ex["category"] == "lookup"]
            
        return examples

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

        state_mapping = {
            "florida": "FL", "california": "CA", "new york": "NY", "texas": "TX",
            "georgia": "GA", "arizona": "AZ", "pennsylvania": "PA", "illinois": "IL",
            "north carolina": "NC", "ohio": "OH"
        }
        for s_name, s_code in state_mapping.items():
            if re.search(r'\b' + s_name + r'\b', q_lower):
                filters["State"] = s_code
                break

        if "State" not in filters:
            state_match = re.search(r'\b(FL|CA|NY|TX|GA|AZ|PA|IL|NC|OH)\b', query.upper())
            if state_match:
                filters["State"] = state_match.group(1)
            
        return filters

    def generate_sql(self, query: str) -> dict:
        """
        Builds Vectorless context and prompts the LLM to generate the final queries.
        """
        logger.info(f"Start planning SQL queries for User Query: '{query}'")
        
        # 1. Retrieve value alignments
        value_alignments = self.lookup_database_values(query)
        if value_alignments:
            logger.info(f"Identified database value alignments: {json.dumps(value_alignments)}")
        
        # 2. Retrieve few-shot templates
        few_shots = self.get_few_shot_examples(query)
        
        # 3. Format value lookup context
        alignment_context = ""
        if value_alignments:
            alignment_context = "\nIdentified value matches in database:\n"
            for val in value_alignments:
                alignment_context += f"- Search term '{val['search_term']}' matched column '{val['matched_column']}' with exact value: '{val['matched_value']}'\n"
        
        # 4. Formulate the prompt
        prompt = f"""You are a precise, schema-aware SQLite SQL Generator for a business intelligence database.
Your job is to translate user natural language questions into valid SQLite SELECT statements.

Schema details for table 'cases':
{json.dumps(TABLE_SCHEMA, indent=2)}
{alignment_context}
Few-shot examples matching user query intent:
{json.dumps(few_shots, indent=2)}

SQL GENERATION RULES:
1. Query Intent & Aggregations:
   - If the user asks for count, volume, frequency, or "how many", use COUNT(*).
   - If the user asks for distributions or breakdowns (e.g. by state, chapter, status, client), group by that column and select COUNT(*) as the aggregate measure (e.g., SELECT State, COUNT(*) as Total_Filings FROM cases GROUP BY State).
2. Thresholds & Filtering Aggregates (HAVING clause):
   - If the user filters on aggregate counts (e.g., "filings greater than 1000", "more than 500 cases"), you MUST use a HAVING clause (e.g. GROUP BY State HAVING COUNT(*) > 1000). Do NOT use WHERE for aggregate constraints.
3. Superlatives, Sorting & Limits:
   - If the user asks for "top N" or "highest N" (e.g., "Which Top 3 states have the highest number of filings?"), group by the dimension, order by the count/aggregation in descending order (e.g., ORDER BY Total_Filings DESC), and apply the exact LIMIT clause (e.g., LIMIT 3).
   - If they ask for "least", "lowest", or "bottom N", order by count/aggregation ascending and apply the LIMIT.
   - Do NOT apply a LIMIT clause if the query asks to filter by thresholds (e.g. "greater than 1000") or requests all matching groups, unless a specific LIMIT or Top N is explicitly requested by the user.
4. Record Selection vs. Aggregations:
   - Do NOT select individual case records (like Ac_no, First_name, Last_name) when the question is an analytical query asking for aggregates, totals, or breakdowns.
   - Select individual records ONLY when the query is a direct record lookup (e.g., showing details, finding a specific debtor, listing pro se cases, showing case status). Apply LIMIT 10 to record lookups to prevent performance degradation.
5. SQLite Specifics:
   - Use SQLite compatible functions (e.g., use strftime('%Y', date_filed) to extract years, rather than YEAR() or EXTRACT()).
   - Use CAST(match_score AS INTEGER) if ordering or filtering by match score numerically.
6. Formatting:
   - Return ONLY a valid JSON list of query configurations. Do not include markdown code wrappers (like ```json), explanations, or notes outside the JSON list.

Output Format:
[
  {{
    "description": "Short explanation of this query's purpose",
    "sql": "SELECT ... FROM cases WHERE ...;"
  }}
]

User Question: {query}
JSON Output:"""

        # Call the LLM
        response = config.call_llm(prompt, system_prompt="You are a JSON-only SQLite SQL generator.")
        
        # Parse the JSON response
        queries_plan = []
        parsed_successfully = False
        if response:
            try:
                cleaned = re.sub(r'```json|```', '', response).strip()
                queries_plan = json.loads(cleaned)
                if isinstance(queries_plan, list) and len(queries_plan) > 0:
                    parsed_successfully = True
            except Exception:
                pass
                
        if not parsed_successfully:
            q_lower = query.lower()
            states_found = re.findall(r'\b(FL|CA|NY|TX|GA|AZ|PA|IL|NC|OH)\b', query.upper())
            chapters_found = re.findall(r'\b(?:chapter\s*)?(7|11|13)\b', q_lower)
            chapters_found = list(dict.fromkeys(chapters_found))
            
            if ("compare" in q_lower or " vs " in q_lower or "versus" in q_lower) and len(states_found) >= 2:
                states_str = ", ".join([f"'{s}'" for s in states_found[:2]])
                queries_plan = [
                    {
                        "description": f"Total filing counts for {', '.join(states_found[:2])}",
                        "sql": f"SELECT State, COUNT(*) as Total_Filings FROM cases WHERE State IN ({states_str}) GROUP BY State;"
                    },
                    {
                        "description": f"Filing counts by Chapter for {', '.join(states_found[:2])}",
                        "sql": f"SELECT State, chapter as Chapter, COUNT(*) as Count FROM cases WHERE State IN ({states_str}) GROUP BY State, Chapter ORDER BY State, Chapter;"
                    },
                    {
                        "description": f"Filing status comparison for {', '.join(states_found[:2])}",
                        "sql": f"SELECT State, status as Status, COUNT(*) as Count FROM cases WHERE State IN ({states_str}) GROUP BY State, Status ORDER BY State, Count DESC;"
                    }
                ]
            elif ("compare" in q_lower or " vs " in q_lower or "versus" in q_lower) and len(chapters_found) >= 2:
                chapters_str = ", ".join([f"'{c}'" for c in chapters_found[:2]])
                queries_plan = [
                    {
                        "description": f"Total filing counts for Chapters {', '.join(chapters_found[:2])}",
                        "sql": f"SELECT chapter as Chapter, COUNT(*) as Total_Filings FROM cases WHERE chapter IN ({chapters_str}) GROUP BY chapter;"
                    },
                    {
                        "description": f"Filing counts by State for Chapters {', '.join(chapters_found[:2])}",
                        "sql": f"SELECT chapter as Chapter, State, COUNT(*) as Count FROM cases WHERE chapter IN ({chapters_str}) GROUP BY chapter, State ORDER BY chapter, Count DESC;"
                    },
                    {
                        "description": f"Filing status comparison for Chapters {', '.join(chapters_found[:2])}",
                        "sql": f"SELECT chapter as Chapter, status as Status, COUNT(*) as Count FROM cases WHERE chapter IN ({chapters_str}) GROUP BY chapter, Status ORDER BY chapter, Count DESC;"
                    }
                ]
            else:
                # If LLM failed, build dynamic smart fallback query matching parsed filters
                filters = self.parse_intent(query)
                is_count_query = any(kw in q_lower for kw in ["count", "how many", "volume", "total number", "number of"])
                is_yearwise = any(kw in q_lower for kw in ["year wise", "year-wise", "yearwise", "yearly", "by year", "over years", "trend by year"])
                is_monthwise = any(kw in q_lower for kw in ["month wise", "month-wise", "monthwise", "monthly", "by month", "trend by month"])
                is_unique = "unique" in q_lower or "distinct" in q_lower
                
                grouping_keywords = ["highest", "most", "distribution", "breakdown", "rank", "volume", "filings", "top", "unique", "distinct", "what are"]
                
                is_state_group = any(w in q_lower for w in ["state", "states"]) and (any(w in q_lower for w in grouping_keywords) or "by state" in q_lower)
                is_chapter_group = any(w in q_lower for w in ["chapter", "chapters"]) and (any(w in q_lower for w in grouping_keywords) or "by chapter" in q_lower)
                is_status_group = any(w in q_lower for w in ["status", "statuses"]) and (any(w in q_lower for w in grouping_keywords) or "by status" in q_lower)
                is_client_group = any(w in q_lower for w in ["client", "clients"]) and (any(w in q_lower for w in grouping_keywords) or "by client" in q_lower)
                is_attorney_group = any(w in q_lower for w in ["attorney", "attorneys", "lawyer", "lawyers", "counsel", "counsels"]) and (any(w in q_lower for w in grouping_keywords) or "by attorney" in q_lower or "handle the most" in q_lower or "handle most" in q_lower or "most cases" in q_lower)
                
                # Exclude lookups from being misclassified as groups
                is_lookup = any(w in q_lower for w in ["record", "records", "case", "cases", "debtor", "debtors", "attorney", "trustee", "judge", "details", "lookup", "retrieve", "who", "whom", "showing"])
                is_group_override = any(w in q_lower for w in ["group by", "by state", "by chapter", "by status", "by client", "by attorney", "breakdown", "distribution", "trend", "count of", "how many", "handle the most", "handle most", "most cases"])
                if is_lookup and not is_group_override:
                    is_state_group = False
                    is_chapter_group = False
                    is_status_group = False
                    is_client_group = False
                    is_attorney_group = False
                
                # Check for threshold criteria (HAVING clause)
                having_clause = ""
                having_desc_suffix = ""
                threshold_match = re.search(r'(?:greater than|more than|>|above)\s*(\d+)', q_lower)
                if threshold_match:
                    threshold_val = threshold_match.group(1)
                    having_clause = f" HAVING COUNT(*) > {threshold_val}"
                    having_desc_suffix = f" with filings > {threshold_val}"
                else:
                    threshold_match_less = re.search(r'(?:less than|fewer than|<|below)\s*(\d+)', q_lower)
                    if threshold_match_less:
                        threshold_val = threshold_match_less.group(1)
                        having_clause = f" HAVING COUNT(*) < {threshold_val}"
                        having_desc_suffix = f" with filings < {threshold_val}"

                limit_match = re.search(r'\b(?:top|highest|lowest|first|last|limit)\s*(\d+)\b', q_lower)
                limit_val = int(limit_match.group(1)) if limit_match else None
                
                # LIMIT clause conditional application:
                # - Use specified limit if limit_val is present.
                # - Default to LIMIT 10 only if there is NO having_clause and NO limit_val (to keep standard breakdowns readable).
                # - Do NOT use any limit if having_clause is present and no explicit limit_val is specified.
                state_limit_str = ""
                if limit_val is not None:
                    state_limit_str = f" LIMIT {limit_val}"
                elif not having_clause:
                    state_limit_str = " LIMIT 10"

                client_limit_str = ""
                if limit_val is not None:
                    client_limit_str = f" LIMIT {limit_val}"
                elif not having_clause:
                    client_limit_str = " LIMIT 10"

                chapter_limit_str = ""
                if limit_val is not None:
                    chapter_limit_str = f" LIMIT {limit_val}"
                elif not having_clause:
                    chapter_limit_str = " LIMIT 10"

                status_limit_str = ""
                if limit_val is not None:
                    status_limit_str = f" LIMIT {limit_val}"
                elif not having_clause:
                    status_limit_str = " LIMIT 10"

                attorney_limit_str = ""
                if limit_val is not None:
                    attorney_limit_str = f" LIMIT {limit_val}"
                elif not having_clause:
                    attorney_limit_str = " LIMIT 10"

                if is_count_query and is_unique:
                    # Determine distinct column
                    distinct_col = None
                    distinct_alias = "Unique_Values"
                    if any(w in q_lower for w in ["client", "clients"]):
                        distinct_col = "client"
                        distinct_alias = "Unique_Clients"
                    elif any(w in q_lower for w in ["state", "states"]):
                        distinct_col = "State"
                        distinct_alias = "Unique_States"
                    elif any(w in q_lower for w in ["chapter", "chapters"]):
                        distinct_col = "chapter"
                        distinct_alias = "Unique_Chapters"
                    elif any(w in q_lower for w in ["status", "statuses"]):
                        distinct_col = "status"
                        distinct_alias = "Unique_Statuses"
                        
                    if distinct_col:
                        if filters:
                            where_clauses = [f'"{k}" = \'{v}\'' for k, v in filters.items()]
                            sql = f"SELECT COUNT(DISTINCT {distinct_col}) as {distinct_alias} FROM cases WHERE {' AND '.join(where_clauses)};"
                            desc = f"Count of unique {distinct_col}s matching filters: {filters}"
                        else:
                            sql = f"SELECT COUNT(DISTINCT {distinct_col}) as {distinct_alias} FROM cases;"
                            desc = f"Count of unique {distinct_col}s in database"
                        queries_plan = [{
                            "description": desc,
                            "sql": sql
                        }]
                    else:
                        # Fallback count
                        if filters:
                            where_clauses = [f'"{k}" = \'{v}\'' for k, v in filters.items()]
                            sql = f"SELECT COUNT(*) as Total_Cases FROM cases WHERE {' AND '.join(where_clauses)};"
                            desc = f"Total case count matching filters: {filters}"
                        else:
                            sql = "SELECT COUNT(*) as Total_Cases FROM cases;"
                            desc = "Total cases in database"
                        queries_plan = [{
                            "description": desc,
                            "sql": sql
                        }]
                elif filters:
                    where_clauses = [f'"{k}" = \'{v}\'' for k, v in filters.items()]
                    if is_state_group:
                        sql = f"SELECT State, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} GROUP BY State{having_clause} ORDER BY Total_Filings DESC{state_limit_str};"
                        desc = f"Top {limit_val} State-wise distribution matching filters: {filters}{having_desc_suffix}" if limit_val else f"State-wise distribution matching filters: {filters}{having_desc_suffix}"
                    elif is_chapter_group:
                        sql = f"SELECT chapter as Chapter, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} GROUP BY Chapter{having_clause} ORDER BY Total_Filings DESC{chapter_limit_str};"
                        desc = f"Top {limit_val} Chapter distribution matching filters: {filters}{having_desc_suffix}" if limit_val else f"Chapter distribution matching filters: {filters}{having_desc_suffix}"
                    elif is_status_group:
                        sql = f"SELECT status as Status, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} GROUP BY Status{having_clause} ORDER BY Total_Filings DESC{status_limit_str};"
                        desc = f"Top {limit_val} Status distribution matching filters: {filters}{having_desc_suffix}" if limit_val else f"Status distribution matching filters: {filters}{having_desc_suffix}"
                    elif is_client_group:
                        sql = f"SELECT client as Client, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} GROUP BY Client{having_clause} ORDER BY Total_Filings DESC{client_limit_str};"
                        desc = f"Top {limit_val} Client distribution matching filters: {filters}{having_desc_suffix}" if limit_val else f"Client distribution matching filters: {filters}{having_desc_suffix}"
                    elif is_attorney_group:
                        where_clauses.append("TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) != ''")
                        sql = f"SELECT TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) as Attorney, COUNT(*) as Total_Cases FROM cases WHERE {' AND '.join(where_clauses)} GROUP BY Attorney{having_clause} ORDER BY Total_Cases DESC{attorney_limit_str};"
                        desc = f"Top {limit_val} Attorneys handling the most cases matching filters: {filters}{having_desc_suffix}" if limit_val else f"Attorneys handling the most cases matching filters: {filters}{having_desc_suffix}"
                    elif is_yearwise:
                        sql = f"SELECT strftime('%Y', date_filed) as Year, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} AND date_filed IS NOT NULL GROUP BY Year ORDER BY Year;"
                        desc = f"Year-wise distribution matching filters: {filters}"
                    elif is_monthwise:
                        sql = f"SELECT strftime('%Y-%m', date_filed) as Month, COUNT(*) as Total_Filings FROM cases WHERE {' AND '.join(where_clauses)} AND date_filed IS NOT NULL GROUP BY Month ORDER BY Month;"
                        desc = f"Month-wise distribution matching filters: {filters}"
                    elif is_count_query:
                        sql = f"SELECT COUNT(*) as Total_Cases FROM cases WHERE {' AND '.join(where_clauses)};"
                        desc = f"Total case count matching filters: {filters}"
                    else:
                        is_trend_query = any(w in q_lower for w in ["trend", "timeline", "time", "date", "history", "chart", "line"])
                        select_cols = "Ac_no as Account_No, case_no as Case_No, First_name as First_Name, Last_name as Last_Name, State, chapter as Chapter, status as Status"
                        if is_trend_query or "date_filed" in q_lower:
                            select_cols += ", date_filed as Date_Filed"
                        order_by_clause = " ORDER BY date_filed ASC" if (is_trend_query or "date_filed" in q_lower) else ""
                        limit_num = limit_val if limit_val is not None else 10
                        sql = f"SELECT {select_cols} FROM cases WHERE {' AND '.join(where_clauses)}{order_by_clause} LIMIT {limit_num};"
                        desc = f"Top {limit_num} Lookup cases matching filters: {filters}" if limit_val else f"Lookup cases matching filters: {filters}"
                    
                    queries_plan = [{
                        "description": desc,
                        "sql": sql
                    }]
                elif is_state_group:
                    queries_plan = [{
                        "description": f"Top {limit_val} State-wise distribution of filings{having_desc_suffix}" if limit_val else f"State-wise distribution of filings{having_desc_suffix}",
                        "sql": f"SELECT State, COUNT(*) as Total_Filings FROM cases GROUP BY State{having_clause} ORDER BY Total_Filings DESC{state_limit_str};"
                    }]
                elif is_chapter_group:
                    queries_plan = [{
                        "description": f"Top {limit_val} Chapter distribution of filings{having_desc_suffix}" if limit_val else f"Chapter distribution of filings{having_desc_suffix}",
                        "sql": f"SELECT chapter as Chapter, COUNT(*) as Total_Filings FROM cases GROUP BY Chapter{having_clause} ORDER BY Total_Filings DESC{chapter_limit_str};"
                    }]
                elif is_status_group:
                    queries_plan = [{
                        "description": f"Top {limit_val} Status distribution of filings{having_desc_suffix}" if limit_val else f"Status distribution of filings{having_desc_suffix}",
                        "sql": f"SELECT status as Status, COUNT(*) as Total_Filings FROM cases GROUP BY Status{having_clause} ORDER BY Total_Filings DESC{status_limit_str};"
                    }]
                elif is_client_group:
                    queries_plan = [{
                        "description": f"Top {limit_val} Client distribution of filings{having_desc_suffix}" if limit_val else f"Client distribution of filings{having_desc_suffix}",
                        "sql": f"SELECT client as Client, COUNT(*) as Total_Filings FROM cases GROUP BY Client{having_clause} ORDER BY Total_Filings DESC{client_limit_str};"
                    }]
                elif is_attorney_group:
                    queries_plan = [{
                        "description": f"Top {limit_val} Attorneys handling the most cases{having_desc_suffix}" if limit_val else f"Attorneys handling the most cases{having_desc_suffix}",
                        "sql": f"SELECT TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) as Attorney, COUNT(*) as Total_Cases FROM cases WHERE TRIM(COALESCE(Attorny_First_Name, '') || ' ' || COALESCE(Attorny_lastt_Name, '')) != '' GROUP BY Attorney{having_clause} ORDER BY Total_Cases DESC{attorney_limit_str};"
                    }]
                elif is_yearwise:
                    queries_plan = [{
                        "description": "Year-wise distribution of filings",
                        "sql": "SELECT strftime('%Y', date_filed) as Year, COUNT(*) as Total_Filings FROM cases WHERE date_filed IS NOT NULL GROUP BY Year ORDER BY Year;"
                    }]
                elif is_monthwise:
                    queries_plan = [{
                        "description": "Month-wise distribution of filings",
                        "sql": "SELECT strftime('%Y-%m', date_filed) as Month, COUNT(*) as Total_Filings FROM cases WHERE date_filed IS NOT NULL GROUP BY Month ORDER BY Month;"
                    }]
                elif is_count_query:
                    queries_plan = [{
                        "description": "Total cases in database",
                        "sql": "SELECT COUNT(*) as Total_Cases FROM cases;"
                    }]
                else:
                    # FTS fallback search (empty list triggers lexical search fallback in retriever)
                    queries_plan = []
            
        logger.info(f"Successfully generated queries plan: {json.dumps(queries_plan)}")

        # Logging to sql_generation.log
        try:
            import os
            import time
            log_path = os.path.join(os.path.dirname(__file__), "sql_generation.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                lf.write(f"[{timestamp}] USER QUERY: {query}\n")
                if value_alignments:
                    lf.write(f"[{timestamp}] VALUE ALIGNMENTS: {json.dumps(value_alignments)}\n")
                for idx, q_item in enumerate(queries_plan, 1):
                    desc = q_item.get("description", "Query")
                    sql = q_item.get("sql", "")
                    lf.write(f"[{timestamp}] SQL {idx} ({desc}):\n{sql}\n")
                lf.write("-" * 80 + "\n")
        except Exception:
            pass

        return {
            "user_query": query,
            "value_alignments": value_alignments,
            "queries": queries_plan
        }

if __name__ == "__main__":
    generator = VectorlessSQLGenerator()
    print("Testing Vectorless RAG value mapping for: 'filings in Jacksonville FL handled by Trustee David'")
    alignments = generator.lookup_database_values("filings in Jacksonville FL handled by Trustee David")
    print(json.dumps(alignments, indent=2))
    
    print("\nGenerating SQL plan...")
    plan = generator.generate_sql("compare filings in TX and AZ")
    print(json.dumps(plan, indent=2))
