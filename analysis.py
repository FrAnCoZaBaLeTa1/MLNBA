import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from model import load_processed_data, create_matchup_features, prepare_training_data, train_baseline_model

def create_baseline_predictions(df):
    """Create baseline predictions using simple rules."""
    # Rule 1: Teams with high win percentage (>0.7) make tournament
    win_pct_threshold = 0.7
    
    # Rule 2: Teams with high BARTHAG (>0.8) make tournament
    barthag_threshold = 0.8
    
    # Rule 3: Teams with high WAB (>3) make tournament
    wab_threshold = 3
    
    # Combine rules
    baseline_preds = (
        (df['WIN_PCT'] > win_pct_threshold) |
        (df['BARTHAG'] > barthag_threshold) |
        (df['WAB'] > wab_threshold)
    ).astype(int)
    
    return baseline_preds

def analyze_errors(model, X_test, y_test, test_data):
    """Analyze model errors and create visualizations."""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create error analysis DataFrame
    error_df = test_data.copy()
    error_df['Prediction'] = y_pred
    error_df['Actual'] = y_test
    error_df['Error'] = y_pred != y_test
    
    # Analyze false positives
    false_positives = error_df[
        (error_df['Prediction'] == 1) & 
        (error_df['Actual'] == 0)
    ]
    
    # Analyze false negatives
    false_negatives = error_df[
        (error_df['Prediction'] == 0) & 
        (error_df['Actual'] == 1)
    ]
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature distributions for errors
    plt.subplot(2, 2, 1)
    sns.boxplot(data=error_df, x='Error', y='WIN_PCT')
    plt.title('Win Percentage Distribution by Error Type')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=error_df, x='Error', y='BARTHAG')
    plt.title('BARTHAG Distribution by Error Type')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=error_df, x='Error', y='WAB')
    plt.title('WAB Distribution by Error Type')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=error_df, x='Error', y='EFF_DIFF')
    plt.title('Efficiency Differential by Error Type')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.close()
    
    return false_positives, false_negatives

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_processed_data()
    matchup_df = create_matchup_features(df)
    X_train, X_test, y_train, y_test = prepare_training_data(matchup_df)
    
    # Get test data for analysis
    test_data = matchup_df[matchup_df['YEAR'].isin(range(2021, 2024))]
    
    # Create baseline predictions
    print("\nCreating baseline predictions...")
    baseline_preds = create_baseline_predictions(test_data)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Train model
    print("\nTraining model...")
    model = train_baseline_model(X_train, y_train)
    model_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {model_accuracy:.4f}")
    
    # Analyze errors
    print("\nAnalyzing model errors...")
    false_positives, false_negatives = analyze_errors(model, X_test, y_test, test_data)
    
    # Print error analysis
    print("\nFalse Positives Analysis:")
    print(false_positives[['TEAM', 'YEAR', 'WIN_PCT', 'BARTHAG', 'WAB']].head())
    
    print("\nFalse Negatives Analysis:")
    print(false_negatives[['TEAM', 'YEAR', 'WIN_PCT', 'BARTHAG', 'WAB']].head())
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main() 