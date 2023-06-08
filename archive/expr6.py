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

    dim = 4
    meanVector = [0]*dim
    # covariance matrix for the errors
    # omega = np.eye(dim)
    omega = np.array([[1.2, 0, 0, 0],
                        [0, 1, 0.4, 0.4],
                        [0, 0.4, 1, 0.3],
                        [0, 0.4, 0.3, 1]])

    W = np.random.multivariate_normal(meanVector, omega, size=size)

    W1 = W[:,0]
    W2 = W[:,1]
    W3 = W[:,2]
    W4 = W[:,3]
    
    A = np.random.binomial(1, expit(0.52 + W1 + W2 + W3 + W4), size)
    if verbose:
       print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.binomial(1, expit(3*A + W2 + W3 + W4), size)
    if verbose:
       print("proportion of Y=1:", np.bincount(Y)[1]/size)
    
    I = np.random.normal(0, 2, size)

    params = [1, 0.5, -1.5]
    pRY_Y_0 = expit(params[0]*W2 + params[0]*W3 + params[0]*W4 + params[1]*I)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "I": I, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "I": I, "R_Y": R_Y})

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

        print("true causal effect", backdoor_adjustment_binary("Y", "A", ["W2", "W3", "W4"], full_data))
        print()

        shadowRecovery = ShadowRecovery("A", "Y", "R_Y", Z, partial_data)
        print("estimated causal effect", shadowRecovery.estimateCausalEffect())
        print("confidence intervals", shadowRecovery.confidenceIntervals())
        print()

        shadowRecoveryIgnoreMissingness = ShadowRecovery("A", "Y", "R_Y", ["W2", "W3", "W4"], subset_data, ignoreMissingness=True)
        print("missingness bias causal effect", shadowRecoveryIgnoreMissingness.estimateCausalEffect())
        print("confidence intervals", shadowRecoveryIgnoreMissingness.confidenceIntervals())
        print()
        print("missingness bias causal effect", backdoor_adjustment_binary("Y", "A", ["W2", "W3", "W4"], subset_data))
        print("confidence intervals", compute_confidence_intervals("Y", "A", ["W2", "W3", "W4"], subset_data, "backdoor_binary"))
        print()

        # shadowRecovery_backdoorBias = ShadowRecovery("A", "Y", "R_Y", ["W1"], partial_data)
        # print("backdoor bias causal effect", shadowRecovery_backdoorBias.estimateCausalEffect())
        # print("confidence intervals", shadowRecovery_backdoorBias.confidenceIntervals())
        # print()
