import pandas as pd
import re
from pathlib import Path
from config.settings import DATA_DIR, PRODUCT_CATEGORIES, TEXT_CLEANING_CONFIG

def load_data(df):
    """Load complaint data from CSV file."""
    file_path = Path(r"C:\Users\Yohanan\updated-intelligent-compliant-analysis\data\compliants.csv")
    df = pd.read_csv(file_path)
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to include only relevant product categories and non-empty narratives."""
    filtered = df[
        (df['Product'].isin(PRODUCT_CATEGORIES)) & 
        (df['Consumer complaint narrative'].notna())
    ].copy()
    return filtered

def clean_text(text: str) -> str:
    """Clean complaint narrative text."""
    if TEXT_CLEANING_CONFIG["lowercase"]:
        text = text.lower()
    
    if TEXT_CLEANING_CONFIG["remove_special_chars"]:
        text = re.sub(r'[^\w\s]', '', text)
    
    if TEXT_CLEANING_CONFIG["remove_boilerplate"]:
        boilerplate_phrases = [
            "i am writing to file a complaint",
            "this is a complaint about",
            "i would like to complain about"
        ]
        for phrase in boilerplate_phrases:
            text = text.replace(phrase, "")
    
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()

def preprocess_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess complaint data."""
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    df['word_count'] = df['cleaned_narrative'].apply(lambda x: len(x.split()))
    return df

def save_processed_data(df: pd.DataFrame, output_file: str):
    """Save processed data to CSV."""
    df.to_csv(Path(DATA_DIR) / output_file, index=False)