"""
Project setup utilities - handles path initialization and project configuration
"""
import os
import sys
from pathlib import Path

def setup_project_paths():
    """
    Initialize project paths for proper imports
    
    This replaces the manual path setup code that was needed in each script.
    Call this once at the beginning of any entry point script.
    Using #1 (Flexible Setup)

    Basic Usage - Manual Setup
    # main.py or any entry point script
    from src.utils.project_setup import setup_project_paths

    # Initialize at the start of your script
    project_root = setup_project_paths()

    # Now you can import your modules and use relative paths
    from src.data.loader import load_data
    from src.models.classifier import MyClassifier

    # Relative paths work because we changed to project root
    data = load_data("data/input.csv")
    Advanced Usage - Full Environment Setup
    # app.py
    from src.utils.project_setup import setup_project_environment

    def main():
        # Complete setup with environment variables
        project_root = setup_project_environment()
        
        # Access environment variable if needed
        import os
        root_from_env = os.environ['PROJECT_ROOT']
        
        # Your application logic here
        from src.api.server import start_server
        start_server()

    if __name__ == "__main__":
        main()
    Conditional Usage - Import Without Auto-Setup
    # If you want to control when setup happens
    import sys
    sys.modules['src.utils.project_setup'] = None  # Prevent auto-setup

    from src.utils.project_setup import setup_project_paths

    # Setup only when you want it
    if some_condition:
        setup_project_paths()


    Returns:
        Path: Project root directory
    """
    
    # Find project root (where this setup is called from)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up from src/utils/
    
    # Alternative: search for marker files
    search_path = current_file.parent
    while search_path != search_path.parent:
        if (search_path / 'pyproject.toml').exists() or (search_path / '.git').exists():
            project_root = search_path
            break
        search_path = search_path.parent
    
    # Add to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Change working directory to project root
    os.chdir(project_root)
    
    print(f"✅ Project initialized: {project_root}")
    return project_root

def setup_project_environment():
    """
    Complete project environment setup
    
    Combines path setup with other initialization tasks
    """
    project_root = setup_project_paths()
    
    # Set environment variables if needed
    os.environ.setdefault('PROJECT_ROOT', str(project_root))
    
    # Initialize any other project-wide settings here
    
    return project_root

# Auto-setup when imported (optional)
if __name__ != "__main__":
    # Only auto-setup if imported, not if run directly
    try:
        setup_project_paths()
    except Exception as e:
        print(f"Warning: Project setup failed: {e}")