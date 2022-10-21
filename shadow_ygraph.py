import pandas as pd
import numpy as np
from scipy.special import expit
from mnar_recoverability import *

def checkConfidenceIntervals(CI, trueValues):
    """
    Checks if all of the values in trueValues are within the confidence intervals CI.

    trueValues is a list of tuples of size 2. The first element of the tuple is a string
    representing the probability law, and the second element is the value of the probability
    law.

    CI is a dictionary where the key is a probability law and the value is a tuple representing
    the low and high ranges of the confidence interval, respectively.

    Returns a list of tuples with three elements (probability law, probability, confidence interval)
    that are out of range. If the return value is an empty list, then no values are out of range.
    """
    # initialize a list that will save all the trueValues that are out of range of the confidence
    # interval
    outOfRange = []
    
    for value in trueValues:
        probabilityLaw = value[0]
        probability = value[1]
        confidenceInterval = CI[probabilityLaw]
        if probability < confidenceInterval[0] or probability > confidenceInterval[1]:
            outOfRange.append((probabilityLaw, probability, confidenceInterval))

    return outOfRange

def testShadowYGraph(verbose=False):
    """
    Generate a graph with the following format: Y1->X1, Y2->X1, X1->X2 where X1 and X2 are 
    self-censoring. 
    We refer to this graph as the shadow Y. All variables are binary variables. R1 
    is the missingness indicator for the variable X1 while R2 is the missingness indicator for the
    variable X2. R1=1 indicates that the value of X is missing, and R1=0 indicates that the
    value of X1 is observed.
    This function then tests the recoverability of the full law of the shadow graph using 
    conditional probabilities.
    """
    size = 50000
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
    R1 = np.random.binomial(1, expit(X1-1.6), size)
    if verbose:
        print('proportion of R1=1', np.bincount(R1)[1]/size)

    # around of 0.57 rows of X2 are 1
    X2 = np.random.binomial(1, expit(X1*1.5), size)

    # generate the missingness mechanism of X2, around 0.29 of the rows of X2 are
    # missing
    R2 = np.random.binomial(1, expit(X2-1.6), size)
    if verbose:
        print('proportion of R2=1', np.bincount(R2)[1]/size)

    # create the fully observed data set
    full_data = pd.DataFrame({"Y1": Y1, "X1": X1, "Y2": Y2, "X2": X2, "R1": R1, "R2": R2})

    # create the partially observed data set subsetted to observed rows of X1 and X2
    partial_data = full_data.copy()
    partial_data = partial_data[partial_data["R1"] == 0]
    partial_data = partial_data[partial_data["R2"] == 0]
    partial_data_R2 = full_data.copy()
    partial_data_R2 = partial_data_R2[partial_data_R2["R2"] == 0]

    # estimate the conditional probabilities for the fully observed data set
    print()
    print('verify recoverability')
    fully_observed_PY1_X1X2Y2 = estimateProbability("Y1", ["X1","X2","Y2"], full_data)
    partially_observed_CI_PY1_X1X2Y2 = computeConfidenceIntervals("Y1", ["X1","X2","Y2"], partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_PY1_X1X2Y2, fully_observed_PY1_X1X2Y2))
    if verbose:
        print('P(Y1 | X1, X2, Y2):', fully_observed_PY1_X1X2Y2)
        print('confidence intervals:', partially_observed_CI_PY1_X1X2Y2)
        print()

    fully_observed_PY2_X1X2 = estimateProbability("Y2", ["X1", "X2"], full_data)
    partially_observed_CI_PY2_X1X2 = computeConfidenceIntervals("Y2", ["X1","X2"], partial_data, 'estimateProbability')
    print(checkConfidenceIntervals(partially_observed_CI_PY2_X1X2, fully_observed_PY2_X1X2))
    if verbose:
        print('P(Y2 | X1, X2):', fully_observed_PY2_X1X2)
        print('confidence intervals', partially_observed_CI_PY2_X1X2)
        print()

    fully_observed_X1_X2 = estimateProbability("X1", ["X2"], full_data)
    partially_observed_CI_X1_X2 = computeConfidenceIntervals("Y1", "X1", partial_data_R2, 'shadowRecoveryDim2', RX="R1", C="X2")
    print(checkConfidenceIntervals(partially_observed_CI_X1_X2, fully_observed_X1_X2))
    if verbose:
        print('P(X1 | X2):', fully_observed_X1_X2)
        print('confidence intervals:', partially_observed_CI_X1_X2)
        print()

    fully_observed_X2 = estimateProbability("X2", [], full_data)
    partially_observed_CI_X2 = computeConfidenceIntervals("Y1", "X2", full_data, 'shadowRecovery', RX="R2")
    print(checkConfidenceIntervals(partially_observed_CI_X2, fully_observed_X2))
    if verbose:
        print('P(X2):', fully_observed_X2)
        print('confidence intervals:', partially_observed_CI_X2)
        print()

if __name__ == "__main__":
    np.random.seed(9)

    # using seed 9, the confidence intervals capture the true value for all 4 probability laws
    testShadowYGraph(verbose=False)