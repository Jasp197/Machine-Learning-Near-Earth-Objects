import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, \
    roc_curve, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Set Pandas options to display numbers with 3-digit decimal precision
pd.set_option('display.float_format', '{:.3f}'.format)

comparison_table = PrettyTable()
comparison_table.title = "Classification techniques comparision"
comparison_table.field_names = ["Algorithm", "Accuracy", "Confusion Matrix", "Precision", "Recall", "Specificity",
                                "F1-score", "ROC-AUC score", "CV Scores"]

random_state = 5805


def format_value(value):
    return "{:.3f}".format(value)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def calculate_metrics(model_name, model, X_test, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Scores for {model_name}: ", format_value(accuracy))
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if model_name != "SVM":
        y_pred_proba = model.predict_proba(X_test)[::, 1]
    else:
        # Calculate decision function for SVM
        y_pred_proba = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=random_state)  # Define the number of splits and random state
    cv_scores = cross_val_score(model, X_test, y_test, cv=skf, scoring='accuracy')

    # Rounding values to 3 decimal places
    cv_scores_new = [round(num, 3) for num in cv_scores]
    print(f"Cross-validated Accuracy Scores: {cv_scores_new}")

    comparison_table.add_row([model_name, format_value(accuracy),
                              conf_matrix, format_value(precision),
                              format_value(recall), format_value(specificity),
                              format_value(f1), format_value(roc_auc), cv_scores_new])

    return fpr, tpr, roc_auc


# ----------------------------------------
#  Phase III: Classification Analysis
# ----------------------------------------


# Pre pruning with grid search parameters
def decisionTreePrePruning(X_train, X_test, y_train, y_test):
    # Define the parameter grid for grid search
    tuned_parameters = {
        'max_depth': range(2, 5, 1),
        'min_samples_split': range(2, 5, 1),
        'min_samples_leaf': range(1, 5, 1),
        'max_features': ['log2', 'sqrt'],
        'splitter': ['best', 'random'],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    decision_tree_model = DecisionTreeClassifier(random_state=random_state)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=decision_tree_model, param_grid=tuned_parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding model
    best_params = grid_search.best_params_
    pre_pruned_tree = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_pre_pruned = pre_pruned_tree.predict(X_test)

    # Display best parameters
    print("\nDecision Tree - Pre Pruned")
    print(f"Best Parameters: {best_params}")

    fpr_pre_pruned, tpr_pre_pruned, roc_auc_pre_pruned = calculate_metrics("Decision Tree - Pre Pruned",
                                                                           pre_pruned_tree, X_test, y_test,
                                                                           y_pred_pre_pruned)

    # Plotting the tree
    plt.figure(figsize=(20, 12))
    tree.plot_tree(pre_pruned_tree, rounded=True, filled=True)
    plt.show()

    print(comparison_table)
    return fpr_pre_pruned, tpr_pre_pruned, roc_auc_pre_pruned, decision_tree_model


# ----------------------------------------


# Post pruning with alpha
def decisionTreePostPruning(X_train, X_test, y_train, y_test, decision_tree_model):
    # Get the effective alphas for different subtrees
    path = decision_tree_model.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    # Plot the accuracy versus effective alpha
    accuracies = []
    for alpha in alphas:
        pruned_tree = DecisionTreeClassifier(random_state=random_state, ccp_alpha=alpha)
        pruned_tree.fit(X_train, y_train)
        y_pred_pruned = pruned_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_pruned)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, accuracies, marker='o', drawstyle='steps-post')
    plt.xlabel("Effective Alpha")
    plt.ylabel("Accuracy on Test Set")
    plt.title("Accuracy vs. Effective Alpha for Post-Pruning")
    plt.grid(True)
    plt.show()

    # Find the optimal alpha that maximizes accuracy
    optimal_alpha = alphas[accuracies.index(max(accuracies))]

    # Create a pruned decision tree with the optimal alpha
    post_pruned_tree = DecisionTreeClassifier(random_state=random_state, ccp_alpha=optimal_alpha)
    post_pruned_tree.fit(X_train, y_train)

    # plotting the tree
    plt.figure(figsize=(20, 12))
    tree.plot_tree(post_pruned_tree, rounded=True, filled=True)
    plt.show()

    # Display the optimal alpha of the pruned tree on the test set
    print("\nDecision Tree - Post Pruned")
    print(f"Optimal Alpha: {optimal_alpha:.6f} (rounded to 6 digit decimal precision)")

    y_pred_post_pruned = post_pruned_tree.predict(X_test)

    fpr_post_pruned, tpr_post_pruned, roc_auc_post_pruned = calculate_metrics("Decision Tree - Post Pruned",
                                                                              post_pruned_tree, X_test, y_test,
                                                                              y_pred_post_pruned)

    # Get the parameters of the optimum tree
    optimum_params = post_pruned_tree.get_params()
    print("Optimum Parameters:")
    for key, value in optimum_params.items():
        if isinstance(value, (float, np.float32, np.float64)):
            formatted_value = "{:.3f}".format(value)
        else:
            formatted_value = value
        print(f"{key}: {formatted_value}")

    print(comparison_table)

    return fpr_post_pruned, tpr_post_pruned, roc_auc_post_pruned


# ----------------------------------------

# Logistic regression
def logisticReg(X_train, X_test, y_train, y_test):
    # Create and train a logistic regression model
    logistic_regression_model = LogisticRegression(random_state=random_state)
    logistic_regression_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_logistic_regression = logistic_regression_model.predict(X_test)

    print("\nLogistic Regression")
    fpr_logistic_regression, tpr_logistic_regression, roc_auc_logistic_regression = calculate_metrics(
        "Logistic Regression", logistic_regression_model, X_test, y_test,
        y_pred_logistic_regression)

    print(comparison_table)

    return fpr_logistic_regression, tpr_logistic_regression, roc_auc_logistic_regression


# ----------------------------------------


# KNN
def knnAnalysis(X_train, X_test, y_train, y_test):
    # Calculating RMSE for different values of K
    rmse_values = []
    k_values = range(1, 21)  # Test K values from 1 to 20

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        rmse = ((y_test - y_pred_knn) ** 2).mean() ** 0.5
        rmse_values.append(rmse)

    # Plotting RMSE values against K values
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, rmse_values, marker='o', linestyle='--')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('RMSE (Root Mean Squared Error)')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    print("\nKNN")

    optimal_k = 11
    print("Optimal k = ", optimal_k)
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    rmse = ((y_test - y_pred_knn) ** 2).mean() ** 0.5
    rmse = format_value(rmse)
    print("RMSE: ", rmse)

    fpr_knn, tpr_knn, roc_auc_knn = calculate_metrics("KNN", knn, X_test, y_test, y_pred_knn)

    print(comparison_table)

    return fpr_knn, tpr_knn, roc_auc_knn


# ----------------------------------------


# SVM
def svmAnalysis(X_train, X_test, y_train, y_test):
    # Parameter grid for different kernels
    param_grid = {
        'linear': {'C': [1, 2, 3]},
        'poly': {'C': [0.1, 1, 2], 'degree': [2, 3, 4]},
        'rbf': {'C': [0.1, 1, 2], 'gamma': ['scale', 'auto']}
    }

    # Initialize the SVM classifiers for each kernel
    svm_linear = SVC(kernel='linear')
    svm_poly = SVC(kernel='poly')
    svm_rbf = SVC(kernel='rbf')

    # Perform GridSearchCV for each SVM classifier
    grid_search_linear = GridSearchCV(svm_linear, param_grid['linear'], cv=2)
    grid_search_linear.fit(X_train, y_train)

    grid_search_poly = GridSearchCV(svm_poly, param_grid['poly'], cv=2)
    grid_search_poly.fit(X_train, y_train)

    grid_search_rbf = GridSearchCV(svm_rbf, param_grid['rbf'], cv=2)
    grid_search_rbf.fit(X_train, y_train)

    print("\nSVM")
    # Best parameters and best scores for each kernel
    print("Best Parameters - Linear Kernel:")
    print(grid_search_linear.best_params_)
    print(f"Best Score: {format_value(grid_search_linear.best_score_)}\n")

    print("Best Parameters - Polynomial Kernel:")
    print(grid_search_poly.best_params_)
    print(f"Best Score: {format_value(grid_search_poly.best_score_)}\n")

    print("Best Parameters - Radial Basis Kernel:")
    print(grid_search_rbf.best_params_)
    print(f"Best Score: {format_value(grid_search_rbf.best_score_)}\n")

    # Linear Kernel SVM
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)

    # Polynomial Kernel SVM
    svm_poly = SVC(kernel='poly')
    svm_poly.fit(X_train, y_train)
    y_pred_poly = svm_poly.predict(X_test)

    # Radial Basis Kernel SVM
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)

    # Accuracy scores for different kernels
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

    print(f"Accuracy Linear Kernel: {format_value(accuracy_linear)}")
    print(f"Accuracy Polynomial Kernel: {format_value(accuracy_poly)}")
    print(f"Accuracy Radial Basis Kernel: {format_value(accuracy_rbf)}")

    print("\nSince Radial Basis Kernel has the highest accuracy, we choose that as our model.\n")
    fpr_svm, tpr_svm, roc_auc_svm = calculate_metrics("SVM", svm_rbf, X_test, y_test, y_pred_rbf)
    print(comparison_table)

    return fpr_svm, tpr_svm, roc_auc_svm


# ----------------------------------------


# Naive Bayes Classifier
def naiveBayes(X_train, X_test, y_train, y_test):
    # Initialize Naive Bayes Classifier
    naive_bayes = GaussianNB()

    # Train the classifier
    naive_bayes.fit(X_train, y_train)

    # Obtain predictions for the test set
    y_pred_nb = naive_bayes.predict(X_test)

    print("\nNaive Bayes")

    fpr_nb, tpr_nb, roc_auc_nb = calculate_metrics("Naive Bayes", naive_bayes, X_test, y_test, y_pred_nb)
    print(comparison_table)

    return fpr_nb, tpr_nb, roc_auc_nb


# ----------------------------------------


# Random Forest Classifier with Grid Search
def randomForest(X_train, X_test, y_train, y_test):
    # Bagging Classifier
    bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=random_state)

    # AdaBoost Classifier
    adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=random_state)

    # Stacking Classifier
    estimators = [('rf', RandomForestClassifier(random_state=random_state)),
                  ('bagging', BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=random_state)),
                  ('adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=random_state))]
    stacking = StackingClassifier(estimators=estimators,
                                  final_estimator=RandomForestClassifier(random_state=random_state))

    param_grid_bagging = {
        'n_estimators': [10, 20],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0]
    }

    param_grid_adaboost = {
        'n_estimators': [20, 30],
        'learning_rate': [0.01, 0.1]
    }

    param_grid_stacking = {
        'final_estimator__n_estimators': [20, 30],
        'rf__max_depth': [None, 5],
        'rf__min_samples_split': [2, 5],
        'bagging__n_estimators': [10, 20],
        'bagging__max_samples': [0.5, 1.0],
        'bagging__max_features': [0.5, 1.0],
        'adaboost__n_estimators': [20, 30],
        'adaboost__learning_rate': [0.01, 0.1]
    }

    # GridSearchCV for each classifier
    grid_bagging = GridSearchCV(bagging, param_grid_bagging, cv=2)
    grid_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=2)
    grid_stacking = GridSearchCV(stacking, param_grid_stacking, cv=2)

    # Fit models
    grid_bagging.fit(X_train, y_train)
    grid_adaboost.fit(X_train, y_train)
    grid_stacking.fit(X_train, y_train)

    # Get best parameters and best scores
    best_params_bagging = grid_bagging.best_params_
    best_score_bagging = grid_bagging.best_score_

    best_params_adaboost = grid_adaboost.best_params_
    best_score_adaboost = grid_adaboost.best_score_

    best_params_stacking = grid_stacking.best_params_
    best_score_stacking = grid_stacking.best_score_

    # Evaluate on test set
    y_pred_bagging = grid_bagging.predict(X_test)
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

    y_pred_adaboost = grid_adaboost.predict(X_test)
    accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)

    y_pred_stacking = grid_stacking.predict(X_test)
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)

    # Print results
    print("\nRandom Forest")

    print("\nBagging - Best Params:", best_params_bagging)
    print("Bagging - Best Score:", format_value(best_score_bagging))
    print("Bagging - Test Accuracy:", format_value(accuracy_bagging))

    print("\nAdaBoost - Best Params:", best_params_adaboost)
    print("AdaBoost - Best Score:", format_value(best_score_adaboost))
    print("AdaBoost - Test Accuracy:", format_value(accuracy_adaboost))

    print("\nStacking - Best Params:", best_params_stacking)
    print("Stacking - Best Score:", format_value(best_score_stacking))
    print("Stacking - Test Accuracy:", format_value(accuracy_stacking))


    print("\nSince BAGGING has the highest accuracy, we choose that as our model.\n")
    fpr_rf, tpr_rf, roc_auc_rf = calculate_metrics("Random Forest - Bagging", grid_bagging, X_test, y_test,
                                                   y_pred_bagging)
    print(comparison_table)

    return fpr_rf, tpr_rf, roc_auc_rf


# ----------------------------------------


# Neural Network
def neuralNetwork(X_train, X_test, y_train, y_test):
    # MLP Classifier with Grid Search
    mlp_classifier = MLPClassifier(random_state=random_state, max_iter=300)

    # Define the parameter grid for MLP
    param_grid_mlp = {
        'hidden_layer_sizes': [(10, 15), (25, 25), (30, 30)],  # hidden layers
        'activation': ['relu', 'tanh'],  # Activation function for hidden layers
        'solver': ['adam', 'sgd'],  # Solver for weight optimization
        'alpha': [0.0001, 0.001, 0.01]  # L2 penalty parameter
    }

    # Perform GridSearchCV for MLP
    grid_search_mlp = GridSearchCV(mlp_classifier, param_grid_mlp, cv=5)
    grid_search_mlp.fit(X_train, y_train)

    print("\nNeural Network")
    print("Best Parameters - MLP Classifier:")
    print(grid_search_mlp.best_params_)
    print(f"Best Score: {format_value(grid_search_mlp.best_score_)}\n")

    best_mlp = grid_search_mlp.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test)

    fpr_mlp, tpr_mlp, roc_auc_mlp = calculate_metrics("Neural Network", best_mlp, X_test, y_test, y_pred_mlp)
    print(comparison_table)

    return fpr_mlp, tpr_mlp, roc_auc_mlp
