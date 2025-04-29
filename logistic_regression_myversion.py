import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# Step 1: Load processed data
print("Loading processed dataset...")
data = pd.read_csv('processed/cbb_cleaned.csv')  # Fixed path to match preprocess.py output
print("Data shape:", data.shape)
print("Data columns:", data.columns.tolist())
print("First few rows:")
print(data.head())

# Step 2: Prepare features and labels
# Using actual columns from the dataset
X = data[['ADJOE', 'ADJDE', 'WIN_PCT', 'BARTHAG', 'TEAM_STRENGTH']]
y = data['POSTSEASON'].apply(lambda x: 1 if x in ['R68', 'R64', 'R32', 'S16', 'E8', 'F4', '2ND', 'Champions'] else 0)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Logistic Regression with SGD
print("Training logistic regression model...")
model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    max_iter=1000,
    learning_rate='constant',
    eta0=0.01,
    random_state=42,
    verbose=1
)
model.fit(X_train_scaled, y_train)

# Step 6: Predictions and evaluation
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Log Loss: {loss:.4f}")

# Step 7: Feature importance analysis
print("\nFeature Importance Analysis:")
feature_names = X.columns
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute Importance': np.abs(coefficients)
}).sort_values('Absolute Importance', ascending=False)

print(feature_importance)

# Step 8: Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Absolute Importance'])
plt.title('Feature Importance in Tournament Prediction')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")

print("Done!")
