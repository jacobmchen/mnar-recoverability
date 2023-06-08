import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize
from adjustment import *
from shadow_recovery import ShadowRecovery
from shadow_covariate_selection import ShadowCovariateSelection
from ratio_test import *

def generateData(size=5000, verbose=False):
    """
    Generate a dataset corresponding to a DAG with the following structure:
    W1->W2, W1->A, W2->A, W2->Y, W2->R_Y, A->Y, A->R_A, Y->R_Y
    W1 and W2 are continuous variables.
    A and Y are binary variables.
    R_A and R_Y are missingness indicators for A and Y, respectively.

    return a 2-tuple: first is a pandas dataframe of the fully observed data,
    second is a pandas dataframe of the partially observed data
    """
    if verbose:
        print(size)

    W1 = np.random.normal(0, 2, size)

    W3 = np.random.normal(0, 2, size)

    W2 = np.random.normal(0, 2, size) + 5*W1 + 5*W3
    
    A = np.random.binomial(1, expit(5*W1), size)
    if verbose:
        print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.normal(0, 2, size) + 2*A + 2.5*W3
    # Y = np.random.binomial(1, expit(5*W3+2*A), size)

    # # R_A=1 denotes observed value of A
    # params = [0.33, -1.71]
    # pRA_A_0 = expit(params[0])
    # R_A_temp = pRA_A_0 / ( pRA_A_0 + np.exp(params[1]*A) * (1-pRA_A_0) )
    # R_A = np.random.binomial(1, R_A_temp, size)
    # if verbose:
    #     print("proportion of R_A=1:", np.bincount(R_A)[1]/size)

    # R_Y=1 denotes observed value of Y
    params = [0.5, -1.4]
    pRY_Y_0 = expit(params[0])
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[1]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_Y": R_Y})

    return (full_data, partial_data)

def generateDataWithU(size=5000, verbose=False):
    """
    Generate a dataset with unmeasured confounding.
    """
    U1 = np.random.normal(0, 1, size)

    W1 = np.random.normal(0, 1, size) + 3*U1

    W2 = np.random.normal(0, 1, size) + 3*U1

    W3 = np.random.normal(0, 1, size) + 3*U1

    U2 = np.random.normal(0, 1, size) + 3*W3

    A = np.random.binomial(1, expit(2.5*W1+ 3*W2), size)
    if verbose:
        print("proportion of A=1:", np.bincount(A)[1]/size)

    # Y = np.random.binomial(1, expit(1.5*U2+3*A), size)
    #if verbose:
    # #    print("proportion of Y=1:", np.bincount(Y)[1]/size)
    Y = np.random.normal(0, 1, size) + 2.5*A + 0.3*W3

    # R_A = np.random.binomial(1, expit(5*W2+5*A), size)
    # if verbose:
    #     print("proportion of R_A=1:", np.bincount(R_A)[1]/size)

    # R_Y = np.random.binomial(1, expit(5*W2+5*Y), size)
    params = [0.5, 0.75, 0.3]
    pRY_Y_0 = expit(params[0]*W2+params[1]*W3)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_Y": R_Y})

    return (full_data, partial_data)

if __name__ == "__main__":
    #np.random.seed(10)

    full_data, partial_data = generateDataWithU(size=50000, verbose=True)

    shadowCovariateSelection = ShadowCovariateSelection("A", "Y", "R_Y", partial_data)
    S, Z = shadowCovariateSelection.findAdjustmentSet()
    print(S, Z)

    # S = "W1"
    # Z = ["W2", "W3"]
    shadowRecovery = ShadowRecovery("A", "Y", "R_Y", Z, S, partial_data)
    print(shadowRecovery.estimateCausalEffect())
    # print(shadowRecovery.confidenceIntervals())

    # subset_data = partial_data[partial_data["R_Y"] == 1]
    # model = sm.GLM.from_formula(formula="Y ~ W1 + W3", data=subset_data, family=sm.families.Gaussian()).fit()
    # print(model.params)
