import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import optimize
from mnar_recoverability import *

def shadowIpwFun(x, Y, X, R, data):
    """
    Defines the shadow IPW recovery function for shadow variable Y, self-censoring variable
    X, and misingness mechanism R using some partially observed dataset data and guess for 
    alpha and gamma values x.
    x is a list of size two where x[0] represents alpha and x[1] represents gamma.
    """
    output1 = np.average( (data[Y]*data[R]) * (expit(x[0]) + (x[1]**data[X])*(1-expit(x[0]))) / expit(x[0]) - data[Y] )
    output2 = np.average( ((1-data[Y])*data[R]) * (expit(x[0]) + (x[1]**data[X])*(1-expit(x[0]))) / expit(x[0]) - (1-data[Y]) )
    
    return [output1, output2]

def computeConfidenceIntervals(Y, X, R, data, value, num_bootstraps=200, alpha=0.05):
    Ql = alpha/2
    Qu = 1 - alpha/2

    estimates = []

    for i in range(num_bootstraps):
        # resample the data with replacement
        bootstrap_data = data.sample(len(data), replace=True)
        bootstrap_data.reset_index(drop=True, inplace=True)

        sol = optimize.root(shadowIpwFun, [0.0, 1.0], args=(Y, X, R, bootstrap_data), method='hybr')

        if value == "X=0":
            estimates.append(expit(sol.x[0]))
        elif value == "X=1":
            estimates.append(expit(sol.x[0])/(expit(sol.x[0]) + sol.x[1]*(1-expit(sol.x[0]))))

    quantiles = np.quantile(estimates, q=[Ql, Qu])

    q_low = quantiles[0]
    q_high = quantiles[1]

    return q_low, q_high

def testShadowGraph(verbose=False):
    """
    Generate a graph with the following format: Y->X->RX. We refer to this graph as the
    shadow graph. Both Y and X are binary variables. RX is the missingness indicator for the
    variable X. RX=1 indicates that the value of X is observed, and RX=0 indicates that the
    value of X is missing.
    This function then recovers the propensity score p(R | X). Once the propensity score
    is recovered, we reweight the dataset using the propensity score to obtain an
    unbiased dataset which then allows us to recover probability distributions as if
    there were no missingness.
    """
    size = 5000
    if verbose:
        print("size:", size)

    # around 0.62 of rows of data of Y are 1
    # Y has no parents, so it does not depend on any other variables
    Y = np.random.binomial(1, 0.62, size)

    # around 0.52 of rows of data of X are 1
    # toggle one of the following DGP processes:
    X = np.random.binomial(1, expit(Y*0.7-0.4), size)
    #X = np.random.binomial(1, 0.52, size)
    
    # generate the missingness mechanism of X, around 0.29 of the rows of X are
    # missing
    # toggle one of the following DGP processes:
    RX = np.random.binomial(1, expit(X+0.5), size)
    #RX = np.random.binomial(1, 0.28, size)
    #RX = np.random.binomial(1, expit(Y-2), size)
    if verbose:
        print('proportion of RX=1', np.bincount(RX)[1]/size)
    # assert that less than 0.3 of the values of X are missing
    assert RX.sum() >= 0.7*size, 'too many missing values in X'

    # create the fully observed data set
    full_data = pd.DataFrame({"Y": Y, "X": X, "RX": RX})

    print("full data set P(R | X)", estimateProbability("RX", ["X"], full_data))

    # create the partially observed data set
    Xstar = X.copy()
    # set missing values to -1 to denote a ? whenever RX == 1
    Xstar[full_data["RX"] == 0] = -1
    obs_data = pd.DataFrame({"Y": Y, "X": Xstar, "RX": RX})
    partial_data = pd.DataFrame({"Y": Y, "X": Xstar, "RX": RX})
    # drop the rows of data where X is unobserved
    partial_data = partial_data[partial_data["X"] != -1]

    # find the roots of the IPW shadow function
    sol = optimize.root(shadowIpwFun, [0.0, 1.0], args=("Y", "X", "RX", obs_data), method='hybr')
    print("P(RX=1 | X=0):", expit(sol.x[0]))
    print("conf intervals", computeConfidenceIntervals("Y", "X", "RX", obs_data, "X=0"))
    print("R(RX=1 | X=1):", expit(sol.x[0])/(expit(sol.x[0]) + sol.x[1]*(1-expit(sol.x[0]))))
    print("conf intervals", computeConfidenceIntervals("Y", "X", "RX", obs_data, "X=1"))

if __name__ == "__main__":
    np.random.seed(10)

    testShadowGraph(verbose=True)