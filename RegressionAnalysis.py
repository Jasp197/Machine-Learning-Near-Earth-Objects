import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

warnings.filterwarnings("ignore")

# Set Pandas options to display numbers with 3-digit decimal precision
pd.set_option('display.float_format', '{:.3f}'.format)


# ----------------------------------------
#  Phase II: Regression Analysis
# ----------------------------------------


# Linear Regression
def LinearRegAlgo(neo_data):
    # Define features and target
    features_linreg = ['est_diameter_min', 'est_diameter_max', 'relative_velocity',
                       'miss_distance', 'orbiting_body', 'sentry_object', 'hazardous']
    target_linreg = 'absolute_magnitude'

    # Create feature matrix and target vector
    X_linreg = neo_data[features_linreg]
    y_linreg = neo_data[target_linreg]

    scaler_linreg = StandardScaler()
    X_linreg_scaled = scaler_linreg.fit_transform(X_linreg)
    X_linreg_scaled_df = pd.DataFrame(X_linreg_scaled, columns=features_linreg)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_linreg_scaled_df, y_linreg, test_size=0.2, random_state=5805)

    results_table = PrettyTable()
    results_table.title = "OLS Results after removing features"
    results_table.field_names = ['Feature Removed', 'AIC', 'BIC', 'Adjusted R-Squared']
    excluded_features = list()

    def ols_dimensionality_red(feature_to_drop):
        excluded_features.append(feature_to_drop)
        X_train_sm.drop(feature_to_drop, axis=1, inplace=True)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        print(model_sm.summary())
        aic = "{:.3f}".format(model_sm.aic)
        bic = "{:.3f}".format(model_sm.bic)
        r_squared_adj = "{:.3f}".format(model_sm.rsquared_adj)
        results_table.add_row([feature_to_drop, aic, bic, r_squared_adj])

    # StatsModels for further statistical measures (AIC, BIC, T-test)
    X_train_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())
    aic = "{:.3f}".format(model_sm.aic)
    bic = "{:.3f}".format(model_sm.bic)
    r_squared_adj = "{:.3f}".format(model_sm.rsquared_adj)
    results_table.add_row(['None', aic, bic, r_squared_adj])

    ols_dimensionality_red('orbiting_body')
    ols_dimensionality_red('sentry_object')
    ols_dimensionality_red('est_diameter_min')
    ols_dimensionality_red('miss_distance')
    ols_dimensionality_red('relative_velocity')
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(results_table)

    # Display the eliminated and final selected features
    OLS_features = X_train_sm.columns[1:].tolist()
    print("Eliminated Features:", excluded_features)
    print("Final Selected Features:", OLS_features)

    # Linear Regression Model
    # Initialize and fit the final model
    X_train_lin = X_train.drop(excluded_features, axis=1)
    X_test_lin = X_test.drop(excluded_features, axis=1)
    model_lin = LinearRegression()
    model_lin.fit(X_train_lin, y_train)
    y_pred_lin = model_lin.predict(X_test_lin)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred_lin)
    print(f"Mean Squared Error (MSE): {mse:.3f}")

    # Plotting the train, test, and predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_lin['est_diameter_max'], y_test, label='Actual', c='blue')
    plt.scatter(X_test_lin['est_diameter_max'], y_pred_lin, label='Predicted', c='red')
    plt.xlabel('Est_diameter_max')
    plt.ylabel('Absolute_magnitude')
    plt.legend()
    plt.title('Linear Regression Predictions')
    plt.show()

    return X_train_lin, X_test_lin, y_train, y_test, model_sm


def PolyRegression(X_train_lin, X_test_lin, y_train, y_test):
    # Polynomial features with degree = 3
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_lin)
    X_test_poly = poly.transform(X_test_lin)

    # Fit a linear regression model (OLS) to the polynomial features
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    # Make predictions on the test set
    y_pred_poly = model_poly.predict(X_test_poly)

    # Calculate Mean Squared Error (MSE)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    print("\nMSE of 3rd order polynomial regression:", "{:.3f}".format(mse_poly))

    # Plotting the train, test, and predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_lin['est_diameter_max'], y_test, label='Actual', c='blue')
    plt.scatter(X_test_lin['est_diameter_max'], y_pred_poly, label='Predicted', c='red')
    plt.xlabel('Est_diameter_max')
    plt.ylabel('Absolute_magnitude')
    plt.legend()
    plt.title('Linear Reg Predictions for polynomial features (degree = 3)')
    plt.show()


def tTest(model_sm):
    # T-test analysis for coefficients
    t_test_results = model_sm.t_test(np.eye(len(model_sm.params)))
    print("\nT-test Results: ")
    print(t_test_results)


def fTest(model_sm):
    # F-test for overall significance
    f_test_results = model_sm.f_test(np.identity(len(model_sm.params)))
    print("\nF-test Results: ")
    print(f_test_results)


def confInterval(model_sm):
    # Confidence interval analysis
    confidence_intervals = model_sm.conf_int()
    print("\nConfidence Intervals: ")
    print(confidence_intervals)
