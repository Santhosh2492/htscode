import os
from dotenv import load_dotenv
from graph import run_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Copy .env.example to .env and add your API key.")

    # Initialize Gemini LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model=os.getenv("HTS_GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=gemini_api_key,
        disable_streaming=True
    )
    
    print("Welcome to the Gemini HTS Code Generator!")
    print("Enter 'quit' to exit.")
    
    while True:
        desc = input("\nEnter item description: ")
        if desc.lower() in ["quit", "exit", "q"]:
            break
            
        print("Running pipeline (Search + RAG)...")
        try:
            result = run_pipeline(desc, gemini_llm)
            print("\n" + "="*50)
            print("REASONING STEPS & CONTEXT")
            print("="*50)
            print(f"Generated Search Queries: {result.get('search_queries', [])}")
            print("\n--- Websites Read (Search Context) ---")
            print(result.get("search_results", "None"))
            print("\n--- Tariff PDF (RAG Context) ---")
            print(result.get("rag_context", "None"))
            print("="*50)
            print(f"\n=> Predicted HTS Code: {result.get('final_hts_code', 'ERROR')}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
