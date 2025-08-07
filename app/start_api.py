#!/usr/bin/env python3
"""
Startup script for the RAG PDF Reader API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"📄 Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
    else:
        print("⚠️ No .env file found. Using system environment variables.")

def check_required_env_vars():
    """Check if required environment variables are set"""
    required_vars = ['API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables or create a .env file.")
        print("You can copy env.example to .env and fill in your values.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting RAG PDF Reader API...")
    print("=" * 50)
    
    # Load environment variables
    load_env_file()
    
    # Check required variables
    if not check_required_env_vars():
        sys.exit(1)
    
    # Import and run the API
    try:
        import uvicorn
        from api import app
        
        print("\n🌐 Starting API server...")
        print("📖 API Documentation will be available at: http://localhost:8000/docs")
        print("🔍 Health check available at: http://localhost:8000/health")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install all required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
