The project starts from FeatureEngineer.py

To run the whole program, please run the following:
    python FeatureEngineering.py

The files needed for this project are:
FeatureEngineering.py
RegressionAnalysis.py
ClassificationAnalysis.py
ClusteringAndAssociation.py

NOTE:
The random forest classifier takes around 30 minutes to run. To directly run the Bagging model without running
the other RF modes, you can replace the code in ClassificationAnalysis.py from lines 351 - 429 with the following:

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
    print("\nRandom Forest")

    print("\nBagging - Best Params:", best_params_bagging)
    print("Bagging - Best Score:", format_value(best_score_bagging))
    print("Bagging - Test Accuracy:", format_value(accuracy_bagging))

