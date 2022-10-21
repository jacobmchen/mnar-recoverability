from functools import partial
import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import optimize
from mnar_recoverability import *
from mnar_ipw_recovery import *
from shadow_ygraph import checkConfidenceIntervals

def findPropensityScore(a, g):
    """
    transform inputs a representing alpha and g representing gamma to p(R=1 | X=0) and
    p(R=1 | X=1), returning the values as a tuple
    """
    return (expit(a), expit(a) / (expit(a) + g*(1-expit(a))))

def testShadowYGraph(verbose=False):
    """
    Generate a graph with the following format: Y1->X1, Y2->X1, X1->X2, Y1->X2 where X1 and X2 are self-censoring. 
    We refer to this graph as the shadow partial Y graph. All variables are binary variables. R1 
    is the missingness indicator for the variable X1 while R2 is the missingness indicator for the
    variable X2. R1=1 indicates that the value of X is observed, and R1=0 indicates that the
    value of X1 is missing.
    This function then uses the fixing technique to reweight the dataset according to the probability
    functions p(R1=1 | X1) and p(R2=1 | X2).
    """
    size = 5000
    if verbose:
        print("size:", size)

    # around 0.62 of rows of data of Y1 are 1
    # Y1 has no parents, so it does not depend on any other variables
    Y1 = np.random.binomial(1, 0.62, size)

    # around 0.54 of rows of Y2 are 1
    # Y2 has no parents, so it does not depend on any other variables
    Y2 = np.random.binomial(1, 0.54, size)

    # around 0.52 of rows of data of X1 are 1
    X1 = np.random.binomial(1, expit(Y1*0.7+Y2-0.7), size)

    # generate the missingness mechanism of X1, around 0.29 of the rows of X1 are
    # missing
    R1 = np.random.binomial(1, expit(X1+0.5), size)
    if verbose:
        print('proportion of R1=1', np.bincount(R1)[1]/size)

    # around of 0.57 rows of X2 are 1
    X2 = np.random.binomial(1, expit(X1*1.5+Y1-1.6), size)

    # generate the missingness mechanism of X2, around 0.29 of the rows of X2 are
    # missing
    R2 = np.random.binomial(1, expit(X2+0.4), size)
    if verbose:
        print('proportion of R2=1', np.bincount(R2)[1]/size)

    # create the fully observed data set
    full_data = pd.DataFrame({"Y1": Y1, "X1": X1, "Y2": Y2, "X2": X2, "R1": R1, "R2": R2})

    if verbose:
        print("full data set P(R1 | X1)", estimateProbability("R1", ["X1"], full_data))
        print("full data set P(R2 | X2)", estimateProbability("R2", ["X2"], full_data))

    X1star = full_data["X1"].copy()
    X1star[full_data["R1"] == 0] = -1
    X2star = full_data["X2"].copy()
    X2star[full_data["R2"] == 0] = -1

    obs_data = pd.DataFrame({"Y1": Y1, "X1": X1star, "Y2": Y2, "X2": X2star, "R1": R1, "R2": R2})

    # find the roots of the shadow IPW function for R1
    roots_R1 = optimize.root(shadowIpwFun, [0.0, 1.0], args=("Y1", "X1", "R1", obs_data), method='hybr')
    propensity_score_R1 = findPropensityScore(roots_R1.x[0], roots_R1.x[1])
    if verbose:
        print("shadow recovery p(R1=1 | X1=0)", propensity_score_R1[0])
        print("shadow recovery p(R1=1 | X1=1)", propensity_score_R1[1])
        print()

    # find the roots of the shadow IPW function for R1
    roots_R2 = optimize.root(shadowIpwFun, [0.0, 1.0], args=("Y1", "X2", "R2", obs_data), method='hybr')
    propensity_score_R2 = findPropensityScore(roots_R2.x[0], roots_R2.x[1])
    if verbose:
        print("shadow recovery p(R2=1 | X2=0)", propensity_score_R2[0])
        print("shadow recovery p(R2=1 | X2=1)", propensity_score_R2[1])
        print()

    # create a dataset where rows of missing data are dropped
    partial_data = obs_data.copy()

    # drop all the rows of data where X1 and X2 are unobserved
    partial_data = partial_data[partial_data["X1"] != -1]
    partial_data = partial_data[partial_data["X2"] != -1]

    scores_R1 = propensity_score_R1[0]**(1-partial_data["X1"]) + propensity_score_R1[1]**(partial_data["X1"]) - 1
    scores_R2 = propensity_score_R2[0]**(1-partial_data["X2"]) + propensity_score_R2[1]**(partial_data["X2"]) - 1

    # create the weights for each row of data by dividing by the propensity score for both missingness
    # indicators, representing fixing in parallel
    partial_data["weights"] = 1 / (scores_R1*scores_R2)

    weighted_partial_data = partial_data.sample(n=len(partial_data), replace=True, weights="weights")

    if verbose:
        print("full data", estimateProbability("X1", [], full_data))
        print("weighted data", estimateProbability("X1", [], weighted_partial_data))
        print("conf intervals", computeConfidenceIntervals("X1", [], weighted_partial_data, "estimateProbability"))
        print()
        print("full data", estimateProbability("X2", [], full_data))
        print("weighted data", estimateProbability("X2", [], weighted_partial_data))
        print("conf intervals", computeConfidenceIntervals("X2", [], weighted_partial_data, "estimateProbability"))
        print()

    print("verify recoverability")
    fully_observed_PY1_X1X2Y2 = estimateProbability("Y1", ["X1","X2","Y2"], full_data)
    partially_observed_CI_PY1_X1X2Y2 = computeConfidenceIntervals("Y1", ["X1","X2","Y2"], weighted_partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_PY1_X1X2Y2, fully_observed_PY1_X1X2Y2))
    if verbose:
        print('P(Y1 | X1, X2, Y2):', fully_observed_PY1_X1X2Y2)
        print('confidence intervals:', partially_observed_CI_PY1_X1X2Y2)
        print()

    fully_observed_PY2_X1X2 = estimateProbability("Y2", ["X1", "X2"], full_data)
    partially_observed_CI_PY2_X1X2 = computeConfidenceIntervals("Y2", ["X1","X2"], weighted_partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_PY2_X1X2, fully_observed_PY2_X1X2))
    if verbose:
        print('P(Y2 | X1, X2):', fully_observed_PY2_X1X2)
        print('confidence intervals', partially_observed_CI_PY2_X1X2)
        print()

    fully_observed_X1_X2 = estimateProbability("X1", ["X2"], full_data)
    partially_observed_CI_X1_X2 = computeConfidenceIntervals("X1", ["X2"], weighted_partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_X1_X2, fully_observed_X1_X2))
    if verbose:
        print('P(X1 | X2):', fully_observed_X1_X2)
        print('confidence intervals:', partially_observed_CI_X1_X2)
        print()

    fully_observed_X2 = estimateProbability("X2", [], full_data)
    partially_observed_CI_X2 = computeConfidenceIntervals("X2", [], weighted_partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_X2, fully_observed_X2))
    if verbose:
        print('P(X2):', fully_observed_X2)
        print('confidence intervals:', partially_observed_CI_X2)
        print()

if __name__ == "__main__":
    np.random.seed(12)

    testShadowYGraph(verbose=False)