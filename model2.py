import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

def load_processed_data():
    """Load the processed dataset."""
    return pd.read_csv('processed/cbb_cleaned.csv')

def create_matchup_features(df):
    """Create features for tournament matchups."""
    # Create a copy of the dataframe
    matchup_df = df.copy()
    
    # Calculate efficiency differentials
    matchup_df['EFF_DIFF'] = matchup_df['ADJOE'] - matchup_df['ADJDE']
    
    # Calculate strength of schedule impact
    matchup_df['SOS_IMPACT'] = matchup_df['BARTHAG'] * matchup_df['WAB']
    
    # Calculate conference strength metrics
    conf_metrics = matchup_df.groupby(['YEAR', 'CONF']).agg({
        'BARTHAG': 'mean',
        'ADJOE': 'mean',
        'ADJDE': 'mean',
        'WAB': 'mean'
    }).reset_index()
    
    conf_metrics.columns = ['YEAR', 'CONF', 'CONF_BARTHAG', 'CONF_ADJOE', 'CONF_ADJDE', 'CONF_WAB']
    matchup_df = matchup_df.merge(conf_metrics, on=['YEAR', 'CONF'])
    
    # Calculate relative conference strength
    matchup_df['REL_CONF_STRENGTH'] = matchup_df['BARTHAG'] / matchup_df['CONF_BARTHAG']
    
    # Calculate recent performance metrics (last 5 games equivalent)
    matchup_df['RECENT_PERF'] = matchup_df['W'] / matchup_df['G']
    
    # Calculate advanced metrics
    matchup_df['OFF_EFF'] = matchup_df['ADJOE'] / matchup_df['CONF_ADJOE']
    matchup_df['DEF_EFF'] = matchup_df['ADJDE'] / matchup_df['CONF_ADJDE']
    
    # Calculate team consistency
    matchup_df['CONSISTENCY'] = matchup_df['WAB'] / matchup_df['G']
    
    # Select features for the model
    features = [
        'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
        'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D',
        '3P_O', '3P_D', 'ADJ_T', 'WAB', 'WIN_PCT', 'EFF_DIFF',
        'SOS_IMPACT', 'CONF_BARTHAG', 'CONF_ADJOE', 'CONF_ADJDE',
        'CONF_WAB', 'REL_CONF_STRENGTH', 'RECENT_PERF', 'OFF_EFF',
        'DEF_EFF', 'CONSISTENCY'
    ]
    
    return matchup_df[features + ['YEAR', 'TEAM', 'POSTSEASON']]

def prepare_training_data(matchup_df, train_years=None, test_years=None):
    """Prepare training and testing data from specific years."""
    if train_years is None:
        train_years = list(range(2013, 2021))  # Use older years for training
    if test_years is None:
        test_years = list(range(2021, 2024))  # Use recent years for testing
    
    # Filter data for training and testing years
    train_data = matchup_df[matchup_df['YEAR'].isin(train_years)]
    test_data = matchup_df[matchup_df['YEAR'].isin(test_years)]
    
    # Features for the model
    features = [
        'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
        'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D',
        '3P_O', '3P_D', 'ADJ_T', 'WAB', 'WIN_PCT', 'EFF_DIFF',
        'SOS_IMPACT', 'CONF_BARTHAG', 'CONF_ADJOE', 'CONF_ADJDE',
        'CONF_WAB', 'REL_CONF_STRENGTH', 'RECENT_PERF', 'OFF_EFF',
        'DEF_EFF', 'CONSISTENCY'
    ]
    
    # Create training and testing sets
    X_train = train_data[features]
    X_test = test_data[features]
    
    # Create target variable (1 if team made tournament, 0 otherwise)
    y_train = (train_data['POSTSEASON'] != 'No Tournament').astype(int)
    y_test = (test_data['POSTSEASON'] != 'No Tournament').astype(int)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=features)
    
    # Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, y_train):
    """Train a baseline random forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance with detailed metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {loss:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, loss

def cross_validate_model(X_train, y_train):
    """Perform cross-validation on the training data."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Use 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print("\nCross-validation scores:")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    print(f"Individual fold scores: {cv_scores}")

def plot_feature_importance(model, features):
    """Plot feature importance from the random forest model."""
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance['Feature'][:10], importance['Importance'][:10])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    print("\nPerforming hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    return grid_search.best_estimator_

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_processed_data()
    matchup_df = create_matchup_features(df)
    
    # Prepare training and testing data
    print("Preparing training and testing data...")
    X_train, X_test, y_train, y_test = prepare_training_data(matchup_df)
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cross_validate_model(X_train, y_train)
    
    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train, y_train)
    
    # Train the model with best parameters
    print("\nTraining model with optimized parameters...")
    best_model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    accuracy, loss = evaluate_model(best_model, X_test, y_test)
    
    # Plot and print feature importance
    print("\nTop 10 Most Important Features:")
    importance = plot_feature_importance(best_model, X_train.columns)
    print(importance.head(10))
    
    # -------- Feature Selection and Retraining -------- #
    print("\nRetraining using top 10 most important features...")

    # Step 1: Select top N features
    top_features = importance.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()

    # Step 2: Subset training and testing data
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    # Step 3: Retrain and evaluate
    selected_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    selected_model.fit(X_train_selected, y_train)

    print("\nEvaluation with Top 10 Features:")
    accuracy_selected, loss_selected = evaluate_model(selected_model, X_test_selected, y_test)

    print(f"\nAccuracy dropped by {accuracy - accuracy_selected:.4f}")
    print(f"Log loss increased by {loss_selected - loss:.4f}")

if __name__ == "__main__":
    main() 
