# src/preprocessing/preprocess.py
import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, brackets, weird symbols, 
    and normalizing whitespace and quotes.
    """
    if pd.isna(text):
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Remove brackets and their contents
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
    
    # Replace fancy quotes with normal quotes
    text = text.replace("``", '"').replace("''", '"').replace("’", "'")
    
    # Remove non-ASCII characters (optional)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe:
    - Dynamically get text columns
    - Clean all text columns
    - Handle missing values
    """
    # Automatically select columns that are likely to contain text
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Clean all text columns
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    
    # Fill missing values with empty string
    df[text_columns] = df[text_columns].fillna("")
    
    return df, text_columns  # return columns for further use (metadata, chunking, etc.)

# Example usage
if __name__ == "__main__":
    from load_data import load_csv, save_csv
    
    # Load CSV dynamically
    df = load_csv("../../data/raw/Natural-Questions-Filtered.csv")
    
    # Show column names and count
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    
    # Preprocess
    df_clean, text_columns = preprocess_dataframe(df)
    
    # Save cleaned CSV
    save_csv(df_clean, "../../data/processed/Natural-Qusitons-Cleaned.csv")
    
    # Print head for inspection
    print(df_clean.head())
