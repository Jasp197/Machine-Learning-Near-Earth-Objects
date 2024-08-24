# Nearest Earth Objects (NEO) Machine Learning Project

This project leverages NASA's Nearest Earth Objects (NEO) dataset to explore and analyze asteroids and other near-Earth objects using various machine learning techniques. The dataset includes attributes such as estimated diameter, relative velocity, miss distance, orbital information, and hazardousness classification. The project involves a comprehensive pipeline of procedures ranging from feature engineering and exploratory data analysis (EDA) to regression, classification, clustering, and association rule mining.

## Project Structure

The project is organized into several Python scripts, each handling a specific phase of the analysis:

- **FeatureEngineering.py**: Performs feature engineering and exploratory data analysis (EDA), including handling missing data, removing duplicates, scaling features, and dimensionality reduction.
- **RegressionAnalysis.py**: Conducts regression analysis, including stepwise regression, linear regression, and polynomial regression, along with T-test, F-test, and confidence interval analysis.
- **ClassificationAnalysis.py**: Implements multiple supervised learning algorithms (Random Forest, SVM, Neural Networks, etc.) for classification tasks, with hyperparameter tuning and performance evaluation metrics.
- **ClusteringAndAssociation.py**: Applies clustering techniques such as K-means and association rule mining using the Apriori algorithm to uncover patterns in the dataset.

## Setup Instructions

### Running the Project

To execute the entire project pipeline, please run the feature engineering file:

python FeatureEngineering.py

The subsequent phases will run automatically.

### Note:

Random Forest Classifier Runtime: The Random Forest classifier may take around 30 minutes to run. If you prefer to skip the Random Forest model and directly run the Bagging model, replace the code in ClassificationAnalysis.py between lines 351 and 429 with the Bagging Classifier code provided in this README.md.

Bagging Classifier Code Snippet:

# Bagging Classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=random_state)

param_grid_bagging = {
    'n_estimators': [10, 20],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0]
}

# GridSearchCV for each classifier
grid_bagging = GridSearchCV(bagging, param_grid_bagging, cv=2)

# Fit models
grid_bagging.fit(X_train, y_train)

# Get best parameters and best scores
best_params_bagging = grid_bagging.best_params_
best_score_bagging = grid_bagging.best_score_

# Evaluate on test set
y_pred_bagging = grid_bagging.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

# Print results
print("\nBagging - Best Params:", best_params_bagging)
print("Bagging - Best Score:", best_score_bagging)
print("Bagging - Test Accuracy:", accuracy_bagging)


Project Highlights

1. Feature Engineering & EDA
Missing Data Handling: Verified no missing data in the dataset.
Outlier Removal: Removed outliers using Z-scores.
Dimensionality Reduction: Applied techniques like Random Forest Analysis, Singular Value Decomposition (SVD), and Principal Component Analysis (PCA).

2. Regression Analysis
Stepwise Regression: Performed to find the best model for predicting the continuous variable absolute_magnitude.
Model Evaluation: Used T-test, F-test, and Confidence Interval Analysis for model validation.

3. Classification Analysis
Algorithms: Employed various algorithms, including Decision Trees, Random Forest, SVM, Neural Networks, and more.
Model Tuning: Hyperparameter tuning via GridSearchCV and model evaluation using metrics such as accuracy, precision, recall, F1-score, and ROC curves.

4. Clustering & Association
K-means Clustering: Identified patterns within the dataset using the K-means++ algorithm.
Association Rule Mining: Applied the Apriori algorithm to uncover associations between features.

Recommendations & Future Work
Feature Selection: From regression analysis, several features (e.g., orbiting_body, sentry_object, est_diameter_min) were deemed unnecessary and can be excluded for better model performance.
Best Classification Model: The Neural Network model emerged as the most effective for classifying hazardous objects.
Clustering: The K-means analysis suggests six distinct clusters in the feature space.
Future Enhancements: Expanding the hyperparameter search space during grid search could potentially improve the model's performance.

Data used -
https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects/data
