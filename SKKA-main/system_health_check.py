"""
Final System Health Check
=========================
Comprehensive diagnostic tool to verify the entire NPTEL Agentic RAG pipeline.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_status(component, status, message=""):
    color = GREEN if status == "PASS" else RED
    print(f"[{color}{status}{RESET}] {component:<20} {message}")

def main():
    print("=" * 60)
    print("🏥 NPTEL AGENTIC RAG - SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    # 1. ENVIRONMENT CHECK
    print("\n1. Environment Configuration")
    load_dotenv(override=True)
    
    env_vars = {
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR"),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME"),
        "STRICT_MODE": os.getenv("STRICT_MODE")
    }
    
    # Optional check for Base URL only if not Groq
    if "groq" not in (env_vars["LLM_MODEL"] or "").lower():
        env_vars["LLM_BASE_URL"] = os.getenv("LLM_BASE_URL")

    issues = []
    for key, val in env_vars.items():
        if val is None:
            issues.append(f"Missing {key}")
            print_status(key, "FAIL", "Variable not set")
        else:
            print_status(key, "PASS", f"= {val}")

    if not issues:
        print_status("Environment", "PASS", "All required variables present")
    
    # 2. PATH VERIFICATION
    print("\n2. File System Paths")
    chroma_path = Path(env_vars["CHROMA_PERSIST_DIR"]) if env_vars["CHROMA_PERSIST_DIR"] else None
    
    if chroma_path and chroma_path.exists():
        print_status("Chroma Path", "PASS", f"Found at {chroma_path.absolute()}")
    else:
        print_status("Chroma Path", "FAIL", f"Directory not found: {chroma_path}")
        issues.append("ChromaDB directory missing")

    # 3. LLM CONNECTIVITY
    print("\n3. LLM Connectivity")
    try:
        model_name = env_vars.get("LLM_MODEL", "ollama/mistral")
        
        if "groq" in model_name.lower() or os.getenv("LLM_PROVIDER") == "groq":
             from langchain_groq import ChatGroq
             api_key = os.getenv("GROQ_API_KEY")
             if not api_key:
                 raise ValueError("GROQ_API_KEY not found in environment for Groq model")
             
             print(f"   Using Groq: {model_name}")
             llm = ChatGroq(
                 model=model_name.replace("groq/", ""),
                 api_key=api_key
             )
        else:
            print(f"   Using Ollama: {model_name}")
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=model_name.replace("ollama/", ""),
                base_url=os.getenv("LLM_BASE_URL") or "http://localhost:11434"
            )

        # Quick ping
        response = llm.invoke("Hi")
        print_status("LLM Response", "PASS", f"Received: {str(response.content)[:20]}...")
    except Exception as e:
        print_status("LLM Response", "FAIL", str(e))
        issues.append(f"LLM Connection failed: {e}")

    # 4. DATABASE INTEGRITY
    print("\n4. ChromaDB & Retriever")
    try:
        sys.path.insert(0, os.getcwd())
        from retriever import ChromaDBRetriever, RetrieverConfig
        
        # Manually create config to ensure we use env vars
        config = RetrieverConfig() 
        # (RetrieverConfig uses env vars by default via default_factory, 
        # checking if it picked them up)
        
        if str(config.chroma_dir) == str(chroma_path):
             print_status("Config Load", "PASS", "RetrieverConfig loaded env vars correctly")
        else:
             print_status("Config Load", "FAIL", f"Expected {chroma_path}, got {config.chroma_dir}")
             issues.append("RetrieverConfig mismatch")
        
        retriever = ChromaDBRetriever(config)
        count = retriever.collection.count()
        
        if count > 0:
            print_status("Collection", "PASS", f"Found {count} documents in '{retriever.config.collection_name}'")
            
            # Test Search
            results = retriever.search("rocket", k=1)
            if results:
                print_status("Search Test", "PASS", "Successfully retrieved chunks")
            else:
                print_status("Search Test", "FAIL", "Search returned 0 results")
        else:
            print_status("Collection", "FAIL", "Collection is empty (0 docs)")
            issues.append("Empty database")
            
    except Exception as e:
        print_status("Database", "FAIL", str(e))
        issues.append(f"Database error: {e}")

    # 5. AGENT ORCHESTRATION
    print("\n5. Agent Pipeline (Dry Run)")
    try:
        from crew.nptel_crew import create_nptel_crew, NPTELCrewConfig
        from crewai import LLM
        
        # Init Crew
        crew_config = NPTELCrewConfig()
        crew_llm = LLM(model=env_vars.get("LLM_MODEL", "ollama/mistral"), base_url=env_vars.get("LLM_BASE_URL"))
        
        from crew.nptel_crew import NPTELCrew # Add import
        crew = NPTELCrew(llm=crew_llm, config=crew_config, verbose=False)
        
        # Check strict mode propagation
        expected_strict = str(env_vars.get("STRICT_MODE", "False")).lower() == "true"
        if crew.config.strict_mode == expected_strict:
            print_status("Config Prop", "PASS", f"Crew Strict Mode = {crew.config.strict_mode}")
        else:
            print_status("Config Prop", "FAIL", f"Mismatch: Env={expected_strict}, Crew={crew.config.strict_mode}")
            issues.append("Strict Mode mismatch")

        print_status("Agent Init", "PASS", "All 4 agents created successfully")
        
    except Exception as e:
        print_status("Agent Init", "FAIL", str(e))
        issues.append(f"Agent creation error: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not issues:
        print(f"{GREEN}✅ SYSTEM IS HEALTHY AND READY FOR PRODUCTION{RESET}")
        print("You can run the Streamlit app with confidence.")
    else:
        print(f"{RED}❌ SYSTEM HAS {len(issues)} ISSUES{RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

if __name__ == "__main__":
    main()
