import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize
from adjustment import *
from shadow_recovery import ShadowRecovery
from shadow_covariate_selection import ShadowCovariateSelection

def generateData(size=5000, verbose=False):
    """
    Generate a dataset.

    return a 2-tuple: first is a pandas dataframe of the fully observed data,
    second is a pandas dataframe of the partially observed data
    """
    if verbose:
        print(size)

    U1 = np.random.normal(0, 1, size)

    U2 = np.random.normal(0, 1, size)

    W3 = np.random.normal(0, 2, size) + U2

    W1 = np.random.normal(0, 2, size) + + 1*U2 + 2*U1

    W2 = np.random.normal(0, 2, size) + 1.5*U1
    
    A = np.random.binomial(1, expit(0.52 + W1 + 3*W3), size)
    if verbose:
       print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.binomial(1, expit(3*A + 3*W2), size)
    if verbose:
       print("proportion of Y=1:", np.bincount(Y)[1]/size)
    
    I = np.random.normal(0, 2, size)

    params = [1, 0.5, -1.5]
    pRY_Y_0 = expit(params[0]*W1 + params[1]*I)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "I": I, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "I": I, "R_Y": R_Y})

    subset_data = partial_data[partial_data["R_Y"] == 1]

    return (full_data, partial_data, subset_data)

if __name__ == "__main__":
    # np.random.seed(0)

    full_data, partial_data, subset_data = generateData(size=10000, verbose=True)
    print()

    shadowCovariateSelection = ShadowCovariateSelection("A", "Y", "R_Y", "I", partial_data)
    result = shadowCovariateSelection.findAdjustmentSet()

    if result is None:
        print("no adjustment set found")
    else:
        W, Z = result

        print("pre-treatment variable:", W)
        print("adjustment set:", Z)

        print("true causal effect", backdoor_adjustment_binary("Y", "A", ["W1", "W2"], full_data))
        print()

        shadowRecovery = ShadowRecovery("A", "Y", "R_Y", Z, partial_data)
        print("estimated causal effect", shadowRecovery.estimateCausalEffect())
        print("confidence intervals", shadowRecovery.confidenceIntervals())
        print()

        print("missingness bias causal effect", backdoor_adjustment_binary("Y", "A", ["W2"], subset_data))
        print("confidence intervals", compute_confidence_intervals("Y", "A", ["W2"], subset_data, "backdoor_binary"))
        print()

        shadowRecovery_backdoorBias = ShadowRecovery("A", "Y", "R_Y", ["W1"], partial_data)
        print("backdoor bias causal effect", shadowRecovery_backdoorBias.estimateCausalEffect())
        print("confidence intervals", shadowRecovery_backdoorBias.confidenceIntervals())
        print()
