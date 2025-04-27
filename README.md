# March Madness Bracket Prediction

This project aims to predict NCAA March Madness tournament participation using machine learning techniques.

## Data Pipeline

### Data Source
The dataset used is `cbb.csv`, containing college basketball statistics from 2013 to 2023. The data includes various team performance metrics such as:
- Offensive and defensive efficiency (ADJOE, ADJDE)
- Strength of schedule (BARTHAG)
- Advanced metrics (EFG%, TO%, etc.)
- Conference information
- Tournament participation

### Preprocessing Steps
1. **Data Cleaning**:
   - Handling missing values
   - Removing duplicates
   - Standardizing data formats

2. **Feature Engineering**:
   - Efficiency differentials (EFF_DIFF)
   - Strength of schedule impact (SOS_IMPACT)
   - Conference strength metrics
   - Recent performance indicators
   - Advanced efficiency metrics
   - Team consistency measures

3. **Data Version Control**:
   - Using DVC to track data and model versions
   - Pipeline configuration in `dvc.yaml`

### Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the preprocessing pipeline:
   ```bash
   dvc repro
   ```

3. Train and evaluate the model:
   ```bash
   python model.py
   ```

## Model Development

### Baseline Model
We initially implemented a Random Forest classifier with default parameters, achieving:
- Accuracy: 93.26%
- Log Loss: 0.1869
- Cross-validation accuracy: 91.67% (±0.37%)

### Hyperparameter Optimization
We performed extensive hyperparameter tuning using GridSearchCV, exploring:
- Number of trees: [50, 100, 200]
- Maximum depth: [5, 10, 15, None]
- Minimum samples per split: [2, 5, 10]
- Minimum samples per leaf: [1, 2, 4]
- Maximum features: ['sqrt', 'log2']

The optimization process:
1. Uses 5-fold cross-validation
2. Evaluates all parameter combinations
3. Selects the best model based on accuracy
4. Retrains the model with optimal parameters

Optimal parameters found:
- max_depth: 15
- max_features: sqrt
- min_samples_leaf: 2
- min_samples_split: 2
- n_estimators: 50

### Model Features
Key features used in the model:
1. Efficiency metrics (ADJOE, ADJDE)
2. Advanced statistics (BARTHAG, EFG%)
3. Conference strength indicators
4. Team performance trends
5. Consistency measures

## Results and Analysis

### Performance Metrics
- Model Accuracy: 92.88%
- Baseline Accuracy: 85.21%
- Log Loss: 0.1871
- Cross-validation accuracy: 91.67% (±0.37%)

### Baseline Comparison
We implemented a simple baseline model using three rules:
1. Teams with win percentage > 0.7
2. Teams with BARTHAG > 0.8
3. Teams with WAB > 3

The baseline achieved 85.21% accuracy, while our model achieved 92.88%, representing a significant 7.67% improvement. This demonstrates the value of our more sophisticated approach.

### Error Analysis
We conducted a detailed analysis of model errors:

1. **False Positives** (Teams predicted to make tournament but didn't):
   - Example: SMU (2021) - High BARTHAG (0.8086) but low WAB (0.2)
   - Example: Belmont (2021) - High win percentage (0.897) but low BARTHAG (0.6786)
   - Common pattern: Teams with strong individual metrics but weak overall tournament profile

2. **False Negatives** (Teams missed by the model):
   - Example: Utah St. (2021) - Good metrics (0.741 win pct, 0.857 BARTHAG) but negative WAB
   - Example: Winthrop (2021) - Excellent win percentage (0.958) but weak schedule
   - Common pattern: Teams with inconsistent metrics across different dimensions

3. **Visual Analysis**:
   - Feature distribution plots show clear patterns in error types
   - Confusion matrix reveals balanced error distribution
   - Error patterns consistent across different metrics

### Feature Importance
Top 5 most important features:
1. Efficiency Differential (EFF_DIFF) - 11.87%
2. Team Consistency - 10.90%
3. Wins Above Bubble (WAB) - 10.47%
4. Strength of Schedule Impact - 7.46%
5. Win Percentage - 6.82%

### Analysis
The optimized model shows:
- Significant 7.67% improvement over baseline
- Better balance between precision and recall
- More robust feature importance distribution
- Reduced false positives (8 vs previous 15)
- Maintained strong overall accuracy

Key findings from error analysis:
1. Most false positives are teams with:
   - High win percentage but weak schedule (e.g., Belmont 2021)
   - Good efficiency but poor tournament metrics (e.g., SMU 2021)
2. Most false negatives are teams with:
   - Strong tournament metrics but inconsistent performance (e.g., Utah St. 2021)
   - Good conference performance but weak overall metrics (e.g., Winthrop 2021)

### Challenges Encountered
1. Initial overfitting with seed and tournament experience features
2. Class imbalance in tournament participation
3. Feature engineering complexity
4. Data quality and completeness issues
5. Hyperparameter optimization computational complexity

## Next Steps

### Short-term Objectives
1. Further analyze error patterns to identify improvement opportunities
2. Implement feature selection based on importance
3. Add more sophisticated features:
   - Head-to-head matchup history
   - Player statistics
   - Injury reports
4. Experiment with ensemble methods
5. Optimize computational efficiency

### Expected Outcomes
1. Further improved prediction accuracy
2. Better handling of edge cases
3. More robust feature importance
4. Enhanced model interpretability
5. Reduced computational complexity
6. Better understanding of error patterns

## Contribution Breakdown

### Omer Yurekli
- Data pipeline implementation
- Feature engineering
- Model development and optimization
- Documentation

### Fanni Soumanou
- developed feature-selected model using top 10 features with evaluation comparison

### Bjorn Shurdha
- documentation
- model analysis
### Franco Zabaleta
- created github repository
- collected the data from various soruces, mainly kaggle
- implemented dvc so that the whole team could readily acceess the data on their home terminal if they wished and so the team could call functions that would manipulate all the data at once and put it into neat graphs that are way better than how appears on cbb files
- Made a graph or two, showing the symmetric nature of our data`
