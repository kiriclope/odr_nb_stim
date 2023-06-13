import statsmodels.api as sm
import pandas as pd
import numpy as np


def glm_abs_error(X_off, X_on):

    X_off_on = np.concatenate((X_off, X_on))

    # OFF = np.hstack((np.ones(X_off.shape[0]), np.zeros(X_on.shape[0])))
    # ON = np.hstack((np.zeros(X_off.shape[0]), np.ones(X_on.shape[0])))

    # Create a dataframe with your data
    conds = np.hstack((np.zeros(X_off.shape[0]), np.ones(X_on.shape[0])))
    # create a dictionary containing the data
    data_dict = {
        "condition": conds,
        # "OFF": OFF,
        # "ON": ON,
        "absolute_errors": X_off_on,
    }

    # create a pandas dataframe from the dictionary
    df = pd.DataFrame(data_dict)

    # Define your formula
    formula = "absolute_errors ~ condition"

    # formula = "absolute_errors ~ OFF + ON + OFF*ON"
    # Fit the generalized linear model with Gaussian family and identity link
    model = sm.formula.ols(formula=formula, data=df).fit()

    # Print the summary of the model
    # print(model.summary())

    return model
