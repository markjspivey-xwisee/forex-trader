import os
import subprocess
import sys

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'data/fetcher.py',
        'data/indicators.py',
        'models/base_model.py',
        'models/random_forest.py',
        'models/neural_network.py',
        'backtesting/backtester.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def check_git():
    """Check if git is initialized and remote is set"""
    try:
        # Check if git is initialized
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        
        # Check if remote exists
        result = subprocess.run(['git', 'remote', '-v'], check=True, capture_output=True, text=True)
        if 'origin' not in result.stdout:
            return False
        return True
    except subprocess.CalledProcessError:
        return False

def setup_streamlit_secrets():
    """Create streamlit secrets file if it doesn't exist"""
    secrets_path = '.streamlit/secrets.toml'
    if not os.path.exists(secrets_path):
        os.makedirs('.streamlit', exist_ok=True)
        with open(secrets_path, 'w') as f:
            f.write("""# Streamlit Secrets
OANDA_API_KEY = "your_api_key"
OANDA_ACCOUNT_ID = "your_account_id"
""")
        print(f"Created {secrets_path} - please update with your API credentials")

def main():
    print("Checking deployment requirements...")
    
    # Check required files
    missing_files = check_requirements()
    if missing_files:
        print("\nError: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    
    # Check git setup
    if not check_git():
        print("\nError: Git repository not initialized or remote not set")
        print("Please run:")
        print("  git init")
        print("  git remote add origin <your-repo-url>")
        sys.exit(1)
    
    # Setup secrets
    setup_streamlit_secrets()
    
    print("\nDeployment checklist:")
    print("✓ All required files present")
    print("✓ Git repository initialized")
    print("✓ Streamlit secrets template created")
    
    print("\nTo deploy to Streamlit Cloud:")
    print("1. Update .streamlit/secrets.toml with your API credentials")
    print("2. Commit your changes:")
    print("   git add .")
    print('   git commit -m "Prepare for deployment"')
    print("3. Push to GitHub:")
    print("   git push origin main")
    print("4. Go to https://streamlit.io/cloud")
    print("5. Connect your GitHub repository")
    print("6. Deploy the app")
    
    print("\nNote: Make sure to set your environment variables in Streamlit Cloud:")
    print("- OANDA_API_KEY")
    print("- OANDA_ACCOUNT_ID")

if __name__ == '__main__':
    main()
