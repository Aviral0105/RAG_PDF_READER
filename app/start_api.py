import os
import sys
import uvicorn
from dotenv import load_dotenv

def run_pre_flight_checks():
    """
    Loads and verifies that necessary environment variables are set.
    """
    print("✈️  Running pre-flight checks...")
    
    # Load variables from .env file into the environment
    load_dotenv()
    
    required_vars = ['API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌  Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("👉  Please create a .env file and add the necessary keys.")
        sys.exit(1) # Exit the script if checks fail
        
    print("✅  All checks passed.")

def main():
    """
    Main function to start the API server for local development.
    """
    run_pre_flight_checks()
    
    print("\n🚀  Starting FastAPI server in development mode...")
    print("🔄  Hot-reloading is enabled.")
    print("📄  API documentation will be available at: http://127.0.0.1:8000/docs")
    print("🛑  Press Ctrl+C to stop the server.\n")
    
    # Programmatically run the Uvicorn server
    uvicorn.run(
        "api:app",          # Path to your FastAPI app instance
        host="127.0.0.1",   # Listen on localhost
        port=8000,          # Use port 8000
        reload=True,        # Enable hot-reloading for development
        log_level="info"    # Set logging level to info
    )

if __name__ == "__main__":
    main()