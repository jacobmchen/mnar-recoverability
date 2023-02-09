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

    W1 = np.random.normal(0, 1, size)

    W3 = np.random.normal(0, 1, size)

    W2 = np.random.normal(0, 1, size) + 1.4*W1 + 2*W3
    
    A = np.random.binomial(1, expit(W1+W2), size)
    if verbose:
        print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.binomial(1, expit(W2+2*A-0.4), size)
    if verbose:
        print("proportion of Y=1:", np.bincount(Y)[1]/size)

    # R_A=1 denotes observed value of A
    params = [0.33, -1.71]
    pRA_A_0 = expit(params[0])
    R_A_temp = pRA_A_0 / ( pRA_A_0 + np.exp(params[1]*A) * (1-pRA_A_0) )
    R_A = np.random.binomial(1, R_A_temp, size)
    if verbose:
        print("proportion of R_A=1:", np.bincount(R_A)[1]/size)

    # R_Y=1 denotes observed value of Y
    params = [0.7, 0.5, -1.4]
    pRY_Y_0 = expit(params[0]+params[1]*W2)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_A": R_A, "R_Y": R_Y})

    # create partially observed dataset
    A = full_data["A"].copy()
    A[full_data["R_A"] == 0] = -1
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "R_A": R_A, "R_Y": R_Y})

    return (full_data, partial_data)

def shadowIpwFunctional_Y(params, W_1, W_2, W_3, R_Y, Y, data):
    """
    Define the system of shadow IPW functions for a shadow variable W_1, a covariate W_2, missingness indicator
    R_Y, and self-censoring outcome variable Y.

    params is a list of four variables alpha_0, alpha_1, gamma_0, and gamma_1 in that order

    return values of the functionals based on the parameters
    """
    # p(R_Y=1 | Y=0)
    pRY_Y_0 = expit(params[0]+params[1]*data[W_2])
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + (np.exp(params[2]*data[Y])) * (1-pRY_Y_0) )

    mean_W1 = np.average(data[W_1])
    var_W1 = np.std(data[W_1])**2

    output1 = np.average( ((mean_W1)*data[R_Y]) / pRY_1 - (mean_W1) )
    output2 = np.average( ((data[W_2])*data[R_Y]) / pRY_1 - (data[W_2]) )
    output3 = np.average( ((var_W1)*data[R_Y]) / pRY_1 - (var_W1) )

    return [output1, output2, output3]

def rootsPrediction_Y(roots, W_2, Y, data):
    """
    Use the values of alpha and gamma contained in roots to make predictions on the value
    Y based on W2.
    """
    pRY_1_Y_0 = expit(roots[0]*data[W_2])
    predictions = pRY_1_Y_0 / (pRY_1_Y_0 + (np.exp(roots[1]*data[Y])) * (1-pRY_1_Y_0))

    return predictions

def shadowIpwFunctional_A(params, W_1, R_A, A, data):
    """
    Define the system of shadow IPW functions for a shadow variable W_1, a missingness indicator
    R_A, and self-censoring outcome variable A.

    params is a list of four variables alpha_0, alpha_1, gamma_0, and gamma_1 in that order

    return values of the functionals based on the parameters
    """
    # p(R_Y=1 | Y=0)
    pRA_A_0 = expit(params[0])
    pRA_1 = pRA_A_0 / (pRA_A_0 + np.exp(params[1]*data[A])*(1-pRA_A_0))

    output1 = np.average( ((data[W_1]*data[R_A])) / pRA_1 - (data[W_1]) )
    output2 = np.average( ((data[W_1]**2)*data[R_A]) / pRA_1 - (data[W_1]**2) )

    return [output1, output2]

def rootsPrediction_A(roots, A, data):
    """
    Use the values of alpha and gamma contained in roots to make predictions on the value
    A.
    """
    pRA_A_0 = expit(roots[0])
    predictions = pRA_A_0 / (pRA_A_0 + np.exp(roots[1]*data[A])*(1-pRA_A_0))

    return predictions

if __name__ == "__main__":
    np.random.seed(10)

    full_data, partial_data = generateData(size=50000, verbose=True)

    # roots_RA = optimize.root(shadowIpwFunctional_A, [0.0, 1.0], args=("W1", "R_A", "A", partial_data), method='hybr')
    # print(roots_RA.x)

    # ##########################
    # # caclulate propensity score of p(R_Y | A, W2)
    # roots_RY = optimize.root(shadowIpwFunctional_Y, [0.0, 0.0, 0.1], args=("W1", "W2", "W3", "R_Y", "Y", partial_data), method='hybr')
    # print(roots_RY.x)

    # # drop rows of data where R_Y=0 and R_A=0
    # subset_data = partial_data[partial_data["R_A"] == 1]
    # subset_data = subset_data[subset_data["R_Y"] == 1]

    # propensityScore_RA = rootsPrediction_A(roots_RA.x, "A", subset_data)
    # propensityScore_RY = rootsPrediction_Y(roots_RY.x, "W2", "Y", subset_data)

    # subset_data["weights"] = 1/(propensityScore_RA*propensityScore_RY)

    # reweight_data = subset_data.sample(n=len(subset_data), replace=True, weights="weights")

    shadowCovariateSelection = ShadowCovariateSelection("A", "R_A", "Y", "R_Y", partial_data)
    print("W1 \\ci R_Y | W2")
    print(shadowCovariateSelection.test_independence("W1", ["W2"]))
    print("W1 \\ci R_Y | W2, A, R_A=1")
    print(shadowCovariateSelection.test_independence("W1", ["W2"], condition_A_R_A=True))
    print(shadowCovariateSelection.findAdjustmentSet())

    # print(backdoor_adjustment_binary("Y", "A", ["W2"], full_data))
    # # print(backdoor_adjustment_binary("Y", "A", ["W2"], reweight_data))
    # # print(compute_confidence_intervals("Y", "A", ["W2"], reweight_data, "backdoor_binary"))

    # shadowRecovery = ShadowRecovery("A", "R_A", "Y", "R_Y", ["W2"], "W1", partial_data)
    # print(shadowRecovery.estimateCausalEffect())
    # print(shadowRecovery.confidenceIntervals())


