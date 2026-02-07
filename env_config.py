from pathlib import Path
from dotenv import load_dotenv


def get_env_file_path():
    """
    Search for .env file starting from the current directory and going up to 4 directory levels.
    
    Checks:
    - Current directory
    - 1 level up
    - 2 levels up
    - 3 levels up
    - 4 levels up
    
    Returns:
        str: The absolute path to the .env file if found
        
    Raises:
        FileNotFoundError: If .env file is not found within 4 directory levels
    """
    # Start from the directory of this file
    current_dir = Path(__file__).parent
    
    # Search up to 4 levels
    for i in range(5):  # 0 to 4 levels (current + 4 up)
        env_path = current_dir / '.env'
        if env_path.exists():
            print(f"Found .env at: {env_path}")
            return str(env_path)
        current_dir = current_dir.parent
    
    raise FileNotFoundError(".env file not found within 4 directory levels")


def load_env():
    """
    Load environment variables from the .env file.
    Automatically finds the .env file using get_env_file_path().
    
    Returns:
        bool: True if .env file exists and was loaded, False otherwise
    """
    try:
        env_path = get_env_file_path()
        return load_dotenv(env_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return False
