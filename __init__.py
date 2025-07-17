# __init__.py (place this in your project root directory)
"""
Document Classifier Project
Automatic path setup for imports
"""
import os
import sys
from pathlib import Path

# Get the project root directory (where this __init__.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add project root to Python path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

# Export for other modules to use
__version__ = "1.0.0"
__author__ = "Woong Choi"

print(f"✅ Document Classifier project initialized")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"🐍 Python path updated")

# Optional: Set environment variables
os.environ.setdefault('PROJECT_ROOT', str(PROJECT_ROOT))