#!/usr/bin/env python3
"""
Environment Setup Script for AI Robotic Car

This script helps users set up their environment variables and test
the configuration for the AI Robotic Car project.

Author: Project Team
License: MIT
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file from template if it doesn't exist"""
    project_root = Path(__file__).parent
    env_template = project_root / 'env.template'
    env_file = project_root / '.env'
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if not env_template.exists():
        print("✗ env.template file not found")
        return False
    
    try:
        # Copy template to .env
        with open(env_template, 'r') as f:
            template_content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(template_content)
        
        print("✓ Created .env file from template")
        print("  Please edit .env file with your actual API keys")
        return True
        
    except Exception as e:
        print(f"✗ Error creating .env file: {e}")
        return False

def check_environment_variables():
    """Check if required environment variables are set"""
    required_vars = ['OPENAI_API_KEY', 'OPENAI_ASSISTANT_ID']
    missing_vars = []
    
    print("Checking environment variables...")
    for var in required_vars:
        if os.getenv(var):
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is not set")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def test_configuration():
    """Test the configuration system"""
    print("\nTesting configuration system...")
    
    try:
        # Add NLP-CV to path
        nlp_cv_path = Path(__file__).parent / 'NLP-CV'
        sys.path.insert(0, str(nlp_cv_path))
        
        from config import validate_required_config, print_config_info
        
        print_config_info()
        
        if validate_required_config():
            print("✓ Configuration is valid")
            return True
        else:
            print("✗ Configuration validation failed")
            return False
            
    except ImportError as e:
        print(f"✗ Could not import configuration: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration test error: {e}")
        return False

def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Set your OpenAI API key and Assistant ID:")
    print("   Option A - Environment variables:")
    print("     export OPENAI_API_KEY='your-api-key'")
    print("     export OPENAI_ASSISTANT_ID='your-assistant-id'")
    print()
    print("   Option B - .env file (recommended):")
    print("     Edit the .env file with your actual API keys")
    print()
    print("   Option C - keys.py file:")
    print("     Copy env.template to keys.py and fill in your keys")
    print()
    print("2. Install dependencies:")
    print("   pip install -r NLP-CV/requirements.txt")
    print()
    print("3. Test the setup:")
    print("   python setup_env.py --test")
    print()
    print("4. Run the application:")
    print("   cd NLP-CV")
    print("   python gpt_car.py")
    print("="*60)

def main():
    """Main setup function"""
    print("AI Robotic Car - Environment Setup")
    print("="*40)
    
    # Check if --test flag is provided
    test_mode = '--test' in sys.argv
    
    # Create .env file if it doesn't exist
    if not test_mode:
        create_env_file()
    
    # Check environment variables
    env_ok = check_environment_variables()
    
    # Test configuration
    config_ok = test_configuration()
    
    # Print results
    print("\n" + "="*40)
    print("SETUP RESULTS")
    print("="*40)
    
    if env_ok and config_ok:
        print("✓ Setup is complete and ready to use!")
        print("You can now run the AI Robotic Car application.")
    else:
        print("✗ Setup is incomplete")
        print_setup_instructions()
    
    print("="*40)

if __name__ == "__main__":
    main() 