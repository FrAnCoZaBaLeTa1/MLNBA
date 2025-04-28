import pandas as pd

print("Loading dataset...")
cbb = pd.read_csv('combined_fixed.csv')

print("Cleaning dataset...")
# Only drop rows if essential columns are missing
essential_columns = ['ADJOE', 'ADJDE', 'WAB', 'BARTHAG', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB']
cbb = cbb.dropna(subset=essential_columns)

print("Generating new features...")
# Calculate additional features safely
cbb['WIN_MARGIN'] = cbb['ADJOE'] - cbb['ADJDE']
cbb['CONSISTENCY_SCORE'] = (cbb['EFG_O'] + cbb['EFG_D']) / 2
cbb['ADJUSTED_WAB'] = cbb['WAB']  # Already exists, rename for clarity if needed
cbb['OFFENSE_POWER'] = cbb['EFG_O'] - cbb['TOR'] + cbb['ORB']
cbb['DEFENSE_POWER'] = cbb['EFG_D'] - cbb['TORD'] + cbb['DRB']

# Optional: filter SEED if SEED exists and reasonable
if 'SEED' in cbb.columns:
    cbb = cbb[cbb['SEED'] <= 16]

print("Saving processed dataset...")
cbb.to_csv('processed_cbb_myversion.csv', index=False)

print("Done! Processed dataset saved as 'processed_cbb_myversion.csv'.")