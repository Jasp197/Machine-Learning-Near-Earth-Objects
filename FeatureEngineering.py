import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import cond
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor

import ClassificationAnalysis
import ClusteringAndAssociation
import RegressionAnalysis

warnings.filterwarnings("ignore")

# Set Pandas options to display numbers with 3-digit decimal precision
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 10)

# File path can be changed according to the location of the neo_v2 dataset
file_path = 'data/neo_v2.csv'

# Read the CSV file using pd.read_csv()
data = pd.read_csv(file_path)
print("Neo Data: \n", data.head())

# ----------------------------------------
# Phase I: Feature Engineering & EDA
# ----------------------------------------


# Total missing data
missing_data = data.isnull().sum()
print("Missing data: \n", missing_data)

# If we had missing values, we can impute them with their mean as follows:

# numerical_cols = data.select_dtypes(include='number').columns
# imputer = SimpleImputer(strategy='mean')
# data[numerical_cols] = imputer.fit_transform(data[numerical_cols])


# Data type of features
data_types = data.dtypes
print("Data type of features: \n", data_types)

# Remove duplicate data
print("Duplicate data: ", data.drop_duplicates(inplace=True))

# Convert boolean features to integer
data['hazardous'] = data['hazardous'].astype(int)
data['sentry_object'] = data['sentry_object'].astype(int)
data['orbiting_body'] = data['orbiting_body'].apply(lambda x: 1 if x == 'Earth' else 0)

# ----------------------------------------

# Outlier removal

columns = ['est_diameter_min', 'est_diameter_max', 'relative_velocity',
           'miss_distance', 'absolute_magnitude']

# Loop through each numerical column and remove outliers based on Z-score
for column in columns:
    z_scores = stats.zscore(data[column])
    threshold = 3  # Set Z-score threshold for outlier detection
    outliers = data[abs(z_scores) > threshold][column]
    data = data[~data[column].isin(outliers)]

# ----------------------------------------

# Random Forest Analysis

# Define features and target
features = ['est_diameter_min', 'est_diameter_max', 'relative_velocity',
            'miss_distance', 'orbiting_body', 'sentry_object', 'absolute_magnitude']
target = 'hazardous'

# Create feature matrix and target vector
X = data[features]
y = data[target]

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=5805)

# Fit the model to the data
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# OBSERVATIONS
# We can see that 'orbiting_body'and 'sentry_object' have no importance, so they can be removed.

# ----------------------------------------

# Singular Value Decomposition

# Standardize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def SVD(n_components):
    svd = TruncatedSVD(n_components=n_components)
    X_svd = svd.fit_transform(X_scaled)

    # Explained variance ratio by the components
    explained_variance_ratio = svd.explained_variance_ratio_
    explained_variance_ratio = [round(num, 3) for num in explained_variance_ratio]

    print(f"Explained variance ratios for {n_components} components: \n", explained_variance_ratio)

print("\nSVD")

num_components = 7  # Number of components to keep
SVD(num_components)

# OBSERVATIONS
# According to SVD, we can remove 3 features without losing much variance

num_components = 4
SVD(num_components)

# ----------------------------------------

# Variance Inflation Factor

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print("\nVIF: \n", vif_data)

# OBSERVATIONS
# Since est_diameter_min and est_diameter_max have inf values, they are highly correlated.
# We can remove one feature and do VIF again

X_vif = X.drop('est_diameter_min', axis=1)
scaler_vif = StandardScaler()
X_vif_scaled = scaler_vif.fit_transform(X_vif)
vif_data2 = pd.DataFrame()
vif_data2["Feature"] = X_vif.columns
vif_data2["VIF"] = [variance_inflation_factor(X_vif_scaled, i) for i in range(X_vif_scaled.shape[1])]
print("\nVIF Final: \n", vif_data2)

# ----------------------------------------

# Condition Number

print("\nPCA and Condition number")

# Calculate the condition number of the data matrix
X_final = X_vif.drop(columns=['orbiting_body', 'sentry_object'])
scaler_cond = StandardScaler()
X_final_scaled = scaler_cond.fit_transform(X_final)
condition_number = cond(X_final_scaled)

# Display the condition number
print(f"Condition number of the data matrix: {condition_number}")

# Principal Component Analysis

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_vif_scaled)

# Percentage of variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio = [round(num, 3) for num in explained_variance_ratio]
print("Explained variance ratio by each component: ", explained_variance_ratio)

# Cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print("Cumulative explained variance ratio: ", cumulative_variance_ratio)

# ----------------------------------------

# One hot encoding SAMPLE
# Our data does not require one hot encoding, but I will provide an example to show how it works

col = ['orbiting_body']

# Using pandas get_dummies to perform one-hot encoding and avoid dummy variable trap
one_hot_encoded = pd.get_dummies(data, columns=col)

# Display the updated dataset after one-hot encoding
print("\nOne Hot Encoding - \n", one_hot_encoded.head())
print("Since we only have one value for Orbiting Body, one hot encoding will create one \n"
      "feature for 'Earth' and assign it the value 1")

# ----------------------------------------

# Covariance Matrix

selected_columns = ['est_diameter_max', 'relative_velocity',
                    'miss_distance', 'absolute_magnitude']

X_final_df = pd.DataFrame(X_final_scaled, columns=selected_columns)

# Calculate the covariance matrix
cov_matrix = X_final_df.cov()

# Plot the heatmap for the covariance matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Covariance Matrix Heatmap')
plt.show()

# ----------------------------------------

# Pearson Correlation Coefficients Matrix

# Calculate the Pearson correlation matrix
corr_matrix = X_final_df.corr()

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Coefficients Heatmap')
plt.show()

# ----------------------------------------

# Balancing the dataset based on target (hazardous)

# Count the occurrences of 0 and 1 in hazardous
target_counts = y.value_counts()

# Display the count of 0 and 1 values
print("\nHazardous values before balancing: \n", target_counts)

# The target is unbalanced. The non hazardous observations are ten times more than hazardous.
# We can do undersampling to balance it.

haz = data[data["hazardous"] == 1]
non_haz = data[data["hazardous"] == 0]

non_haz_downsample = resample(non_haz,
                              replace=False,
                              n_samples=len(haz),
                              random_state=5805)

data_downsampled = pd.concat([non_haz_downsample, haz])
neo_data = data_downsampled.sample(frac=1, random_state=5805).reset_index(drop=True)

print("Hazardous values after balancing: \n", neo_data["hazardous"].value_counts())


# ----------------------------------------
#  Phase II: Regression Analysis
# ----------------------------------------


X_train_lin, X_test_lin, y_train, y_test, model_sm = RegressionAnalysis.LinearRegAlgo(neo_data)
RegressionAnalysis.PolyRegression(X_train_lin, X_test_lin, y_train, y_test)
RegressionAnalysis.tTest(model_sm)
RegressionAnalysis.fTest(model_sm)
RegressionAnalysis.confInterval(model_sm)


# ----------------------------------------
#  Phase III: Classification Analysis
# ----------------------------------------


# Define features and target
features_class = ['est_diameter_max', 'relative_velocity',
                  'miss_distance', 'absolute_magnitude']
target_class = 'hazardous'

# Create feature matrix and target vector
X_class = neo_data[features_class]
y_class = neo_data[target_class]

scaler_class = StandardScaler()
X_class_scaled = scaler_class.fit_transform(X_class)
X_class_scaled_df = pd.DataFrame(X_class_scaled, columns=features_class)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_class_scaled_df, y_class, test_size=0.2,
                                                    random_state=5805)

fpr_pre_pruned, tpr_pre_pruned, roc_auc_pre_pruned, decision_tree_model = ClassificationAnalysis.decisionTreePrePruning(X_train, X_test, y_train, y_test)
fpr_post_pruned, tpr_post_pruned, roc_auc_post_pruned = ClassificationAnalysis.decisionTreePostPruning(X_train, X_test, y_train, y_test, decision_tree_model)
fpr_logistic_regression, tpr_logistic_regression, roc_auc_logistic_regression = ClassificationAnalysis.logisticReg(X_train, X_test, y_train, y_test)
fpr_knn, tpr_knn, roc_auc_knn = ClassificationAnalysis.knnAnalysis(X_train, X_test, y_train, y_test)
fpr_svm, tpr_svm, roc_auc_svm = ClassificationAnalysis.svmAnalysis(X_train, X_test, y_train, y_test)
fpr_nb, tpr_nb, roc_auc_nb = ClassificationAnalysis.naiveBayes(X_train, X_test, y_train, y_test)
fpr_rf, tpr_rf, roc_auc_rf = ClassificationAnalysis.randomForest(X_train, X_test, y_train, y_test)
fpr_mlp, tpr_mlp, roc_auc_mlp = ClassificationAnalysis.neuralNetwork(X_train, X_test, y_train, y_test)

# Create the ROC curve plot comparing all algorithms
plt.figure(figsize=(10, 8))
plt.plot(fpr_pre_pruned, tpr_pre_pruned, lw=2,
         label=f'Pre pruned ROC curve (AUC = {roc_auc_pre_pruned:.3f})')
plt.plot(fpr_post_pruned, tpr_post_pruned, lw=2,
         label=f'Post pruned ROC curve (AUC = {roc_auc_post_pruned:.3f})')
plt.plot(fpr_logistic_regression, tpr_logistic_regression, lw=2,
         label=f'Logistic regression ROC curve (AUC = {roc_auc_logistic_regression:.3f})')
plt.plot(fpr_knn, tpr_knn, lw=2,
         label=f'KNN ROC curve (AUC = {roc_auc_knn:.3f})')
plt.plot(fpr_svm, tpr_svm, lw=2,
         label=f'SVM ROC curve (AUC = {roc_auc_svm:.3f})')
plt.plot(fpr_nb, tpr_nb, lw=2,
         label=f'Naive Bayes ROC curve (AUC = {roc_auc_nb:.3f})')
plt.plot(fpr_rf, tpr_rf, lw=2,
         label=f'Random Forest ROC curve (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_mlp, tpr_mlp, lw=2,
         label=f'MLP ROC curve (AUC = {roc_auc_mlp:.3f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# ----------------------------------------
# Phase IV: Clustering and Association
# ----------------------------------------


# K Means Algorithm
ClusteringAndAssociation.kMeansAlgo(X_class_scaled_df)

# ----------------------------------------

# Apriori Algorithm
ClusteringAndAssociation.aprioriAlgo(X_class_scaled_df)
