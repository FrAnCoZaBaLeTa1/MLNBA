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

## Baseline Model

We implemented a Random Forest classifier as our baseline model with the following specifications:
- 100 trees
- Maximum depth of 10
- Minimum samples per split: 5
- Minimum samples per leaf: 2
- Balanced class weights

### Model Features
Key features used in the model:
1. Efficiency metrics (ADJOE, ADJDE)
2. Advanced statistics (BARTHAG, EFG%)
3. Conference strength indicators
4. Team performance trends
5. Consistency measures

## Preliminary Results

### Performance Metrics
- Accuracy: 93.26%
- Log Loss: 0.1869
- Cross-validation accuracy: 91.67% (±0.37%)

### Feature Importance
Top 5 most important features:
1. Efficiency Differential (EFF_DIFF)
2. Wins Above Bubble (WAB)
3. Team Consistency
4. Recent Performance
5. Strength of Schedule Impact

### Analysis
The model shows strong predictive power, particularly in:
- Identifying strong tournament teams (high precision)
- Maintaining good overall accuracy
- Balancing between precision and recall

### Challenges Encountered
1. Initial overfitting with seed and tournament experience features
2. Class imbalance in tournament participation
3. Feature engineering complexity
4. Data quality and completeness issues

## Next Steps

### Short-term Objectives
1. Implement hyperparameter tuning
2. Add more sophisticated features:
   - Head-to-head matchup history
   - Player statistics
   - Injury reports
3. Experiment with different model architectures

### Expected Outcomes
1. Improved prediction accuracy
2. Better handling of edge cases
3. More robust feature importance
4. Enhanced model interpretability

## Contribution Breakdown

### Omer Yurekli
- Data pipeline implementation
- Feature engineering
- Model development
- Documentation