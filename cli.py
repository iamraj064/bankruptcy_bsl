import sys
import argparse
import json
from vectorless_rag import VectorlessRAGQA

def run_cli():
    parser = argparse.ArgumentParser(description="Vectorless RAG QA CLI for Bankruptcy Database")
    parser.add_argument("--query", "-q", type=str, help="Natural language question to query")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of records to retrieve (default: 5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive command line shell")
    
    args = parser.parse_args()
    qa_engine = VectorlessRAGQA()
    
    if args.query:
        res = qa_engine.generate_answer(args.query, top_k=args.top_k)
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        print("="*80 + "\n")
        print(res["answer"])
        print("\n" + "-"*80)
        print("RETRIEVAL METADATA:")
        print(json.dumps(res["metadata"], indent=2))
        print("="*80 + "\n")
        return

    if args.interactive or not args.query:
        print("\n========================================================")
        print("  Vectorless RAG Question Answering System (CLI)")
        print("  Database: vl_rag.db (10,000 Records | FTS5 BM25 + SQL)")
        print("========================================================\n")
        print("Type your question below (or 'exit' / 'quit' to stop).\n")
        
        while True:
            try:
                user_input = input("Vectorless RAG > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting Vectorless RAG CLI. Goodbye!")
                    break
                    
                res = qa_engine.generate_answer(user_input, top_k=args.top_k)
                print("\n" + "-"*60)
                print(res["answer"])
                print("\n[Metrics] Latency:", f"{res['metadata']['retrieval_time_ms']} ms", "| Mode:", res['metadata']['search_mode'], "| Embeddings: 0")
                print("-" * 60 + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting CLI...")
                break

if __name__ == "__main__":
    run_cli()
