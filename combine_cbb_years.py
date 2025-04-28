import pandas as pd
import os

folder_path = 'archive'
dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename.startswith('cbb'):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV
        df = pd.read_csv(file_path)

        # Extract year from filename safely
        year_part = filename.replace('cbb', '').replace('.csv', '')
        
        if year_part.isdigit():
            year = int(year_part) + 2000
            df['YEAR'] = year
            dfs.append(df)
        else:
            print(f"⚠️ Skipping file with unexpected name: {filename}")

# Combine all good dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv('combined_fixed.csv', index=False)
    print("✅ Done combining all files into combined_fixed.csv")
else:
    print("❌ No valid CSV files found!")