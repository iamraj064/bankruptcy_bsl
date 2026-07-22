import sqlite3
import csv
import time
import os
import re
import logger_config

DB_PATH = os.path.join(os.path.dirname(__file__), "vl_rag.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
logger = logger_config.setup_logger("db_builder")

def clean_val(v):
    if v is None:
        return ""
    return str(v).strip()

def build_database():
    logger.info(f"Starting database build from {CSV_PATH}...")
    start_time = time.time()
    
    if os.path.exists(DB_PATH):
        logger.info(f"Removing existing database at {DB_PATH}")
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Read CSV header and rows
    with open(CSV_PATH, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        
        # Build SQL schema
        col_defs = [f'"{h}" TEXT' for h in headers]
        create_table_sql = f'CREATE TABLE cases ({", ".join(col_defs)});'
        cursor.execute(create_table_sql)
        
        # Create FTS5 virtual table
        # We index all key searchable fields + an 'all_text' field combining everything for lexical search
        fts_cols = [
            "Ac_no", "case_no", "client", "debtor_name", "primary_debtor", 
            "secondary_debtor", "attorney_name", "trustee_name", "Judge_name", 
            "court_district", "Court_id", "State", "City", "chapter", "status", 
            "notice_type", "Disposition_text", "all_text"
        ]
        create_fts_sql = f'''
        CREATE VIRTUAL TABLE cases_fts USING fts5(
            {", ".join(fts_cols)},
            tokenize = 'porter unicode61'
        );
        '''
        cursor.execute(create_fts_sql)
        
        # Prepare insert statements
        placeholders = ", ".join(["?"] * len(headers))
        insert_rel_sql = f'INSERT INTO cases VALUES ({placeholders})'
        insert_fts_sql = f'INSERT INTO cases_fts VALUES ({", ".join(["?"] * len(fts_cols))})'
        
        rel_batch = []
        fts_batch = []
        
        for row_idx, row in enumerate(reader):
            # Map row to dictionary for easy field access
            row_dict = {headers[i]: clean_val(row[i]) if i < len(row) else "" for i in range(len(headers))}
            
            # Extract composite field values
            debtor_name = " ".join(filter(None, [row_dict.get("First_name"), row_dict.get("middle_name"), row_dict.get("Last_name")]))
            pd_name = " ".join(filter(None, [row_dict.get("PD_First_Name"), row_dict.get("PD_middle_Name"), row_dict.get("PD_lastt_Name")]))
            sd_name = " ".join(filter(None, [row_dict.get("SD_First_Name"), row_dict.get("SD_middle_Name"), row_dict.get("SD_lastt_Name")]))
            attorney_name = " ".join(filter(None, [row_dict.get("Attorny_First_Name"), row_dict.get("Attorny_middle_Name"), row_dict.get("Attorny_lastt_Name")]))
            
            all_text_content = " | ".join([f"{k}: {v}" for k, v in row_dict.items() if v])
            
            rel_batch.append([row_dict.get(h, "") for h in headers])
            fts_batch.append([
                row_dict.get("Ac_no", ""),
                row_dict.get("case_no", ""),
                row_dict.get("client", ""),
                debtor_name,
                pd_name,
                sd_name,
                attorney_name,
                row_dict.get("trustee_name", ""),
                row_dict.get("Judge_name", ""),
                row_dict.get("court_district", ""),
                row_dict.get("Court_id", ""),
                row_dict.get("State", ""),
                row_dict.get("City", ""),
                row_dict.get("chapter", ""),
                row_dict.get("status", ""),
                row_dict.get("notice_type", ""),
                row_dict.get("Disposition_text", ""),
                all_text_content
            ])
            
            if len(rel_batch) >= 1000:
                cursor.executemany(insert_rel_sql, rel_batch)
                cursor.executemany(insert_fts_sql, fts_batch)
                rel_batch = []
                fts_batch = []
                
        if rel_batch:
            cursor.executemany(insert_rel_sql, rel_batch)
            cursor.executemany(insert_fts_sql, fts_batch)
            
    # Create indexes on relational table for vectorless high-performance lookup
    indexed_columns = [
        "Ac_no", "case_no", "client", "ssn", "chapter", "status", 
        "State", "City", "Court_State", "Judge_name", "trustee_name", 
        "date_filed", "notice_type"
    ]
    for col in indexed_columns:
        if col in headers:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{col.lower()} ON cases("{col}");')
            
    conn.commit()
    
    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM cases;")
    total_rel = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM cases_fts;")
    total_fts = cursor.fetchone()[0]
    
    conn.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Database build complete in {elapsed:.2f} seconds!")
    logger.info(f"Total Relational Records: {total_rel}")
    logger.info(f"Total FTS5 Text Indexed Records: {total_fts}")
    return total_rel

if __name__ == "__main__":
    build_database()
