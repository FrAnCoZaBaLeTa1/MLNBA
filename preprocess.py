import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load the college basketball dataset from individual year files."""
    archive_path = Path('archive')
    all_data = []
    
    # Load individual year files
    for year_file in sorted(archive_path.glob('cbb[0-9][0-9].csv')):
        year = int(year_file.stem[-2:])  # Extract year from filename
        df = pd.read_csv(year_file)
        df['YEAR'] = 2000 + year  # Convert to full year
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def clean_data(df):
    """Clean the dataset by handling missing values and duplicates."""
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values in SEED and POSTSEASON columns
    # These are expected to be missing for teams that didn't make the tournament
    df['SEED'] = df['SEED'].fillna(0)
    df['POSTSEASON'] = df['POSTSEASON'].fillna('No Tournament')
    
    return df

def engineer_features(df):
    """Engineer new features for the dataset."""
    # Calculate win percentage
    df['WIN_PCT'] = df['W'] / df['G']
    
    # Calculate seed difference (for tournament games)
    # This will be useful when creating matchup features later
    df['SEED_DIFF'] = 0  # Initialize with 0
    
    # Adjusted efficiency metrics are already in the dataset as ADJOE and ADJDE
    
    return df

def save_processed_data(df):
    """Save the processed dataset."""
    output_path = Path('processed/cbb_cleaned.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    # Load the data
    print("Loading data...")
    df = load_data()
    
    # Clean the data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Engineer features
    print("Engineering features...")
    df = engineer_features(df)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(df)
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(df.head())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Display year distribution
    print("\nYear distribution:")
    print(df['YEAR'].value_counts().sort_index())

if __name__ == "__main__":
    main() 