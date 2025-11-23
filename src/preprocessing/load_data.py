# src/preprocessing/load_data.py
import pandas as pd
from pathlib import Path

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into Pandas DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

def save_csv(df: pd.DataFrame, file_path: str):
    """
    Save DataFrame to CSV
    """
    df.to_csv(file_path, index=False)
    print(f"Saved DataFrame to {file_path}")

# Example usage
if __name__ == "__main__":
    df = load_csv("../../data/raw/Natural-Questions-Filtered.csv")
    print(df.head())
