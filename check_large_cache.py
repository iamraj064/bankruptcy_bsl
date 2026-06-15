import sqlite3

def check_cache():
    conn = sqlite3.connect('.litellm_prompt_cache.db')
    c = conn.cursor()
    h = "c71ce16b86b424c0b8637d96466d808cd8ede35601fc68dbe32d9a7558c49458"
    c.execute("SELECT prompt, response FROM prompt_cache WHERE prompt_hash=?", (h,))
    row = c.fetchone()
    if row:
        prompt, response = row
        print("PROMPT PREFIX:")
        print(prompt[:400] + "...")
        print("-" * 40)
        print("RESPONSE LENGTH:", len(response))
        print("RESPONSE:")
        print(response)
    else:
        print("Not found")
    conn.close()

if __name__ == '__main__':
    check_cache()
