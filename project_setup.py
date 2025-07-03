# project_setup.py
import os
import sys
from pathlib import Path

def initialize() -> Path:
    """
    Sets up the project environment by resolving path issues.

    This function finds the project root directory, changes the current
    working directory to it, and adds it to the system path. This
    ensures that both relative file paths and module imports work
    correctly from any script.

    Returns:
        Path: The absolute path to the project root directory.
    """
    # Find the project root by looking for a marker file (e.g., 'pyproject.toml' or '.git')
    # This is more robust than using '__file__' with '..'
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent

    # In case this script is moved, you can search upwards for a marker
    while not (project_root / 'pyproject.toml').exists() and not (project_root / '.git').exists():
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root. Make sure a 'pyproject.toml' or '.git' folder exists.")
        project_root = project_root.parent

    # 1. Fix `FileNotFoundError`: Change the current working directory
    os.chdir(project_root)

    # 2. Fix `ModuleNotFoundError`: Add project root to the import path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"✅ Project environment initialized. Root: {project_root}")
    return project_root

# Automatically initialize when this module is imported
PROJECT_ROOT = initialize()