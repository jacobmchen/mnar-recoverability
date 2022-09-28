import pandas as pd
import numpy as np
from scipy.special import expit
from mnar_recoverability import *

def testShadowYGraph(verbose=False):
    """
    Generate a graph with the following format: Y1->X1->Y2->X2 where X1 and X2 are self-censoring. 
    We refer to this graph as the shadow graph where n=2. All variables are binary variables. R1 
    is the missingness indicator for the variable X1 while R2 is the missingness indicator for the
    variable X2. RX=1 indicates that the value of X is missing, and RX=0 indicates that the
    value of X is observed.
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
    # print('full_data')
    # print(full_data)

    # estimate the conditional probabilities for the fully observed data set
    print()
    print('fully observed data set')
    print('P(Y1 | X1, X2, Y2):', estimateProbability("Y1", ["X1","X2","Y2"], full_data))
    print()
    print('P(Y2 | X1, X2):', estimateProbability("Y2", ["X1", "X2"], full_data))
    print()
    print('P(X1 | X2):', estimateProbability("X1", ["X2"], full_data))
    print()
    print('P(X2):', estimateProbability("X2", [], full_data))


    # create the partially observed data set subsetted to observed rows of X1 and X2
    partial_data = full_data.copy()
    partial_data = partial_data[partial_data["R1"] == 0]
    partial_data = partial_data[partial_data["R2"] == 0]
    partial_data_R2 = full_data.copy()
    partial_data_R2 = partial_data_R2[partial_data_R2["R2"] == 0]

    print()
    print('partially observed data set')
    print('P(Y1 | X1, X2, Y2):', estimateProbability("Y1", ["X1","X2","Y2"], partial_data))
    print('confidence intervals:', computeConfidenceIntervals("Y1", ["X1","X2","Y2"], partial_data, 'estimateProbability'))
    print()
    print('P(Y2 | X1, X2):', estimateProbability("Y2", ["X1", "X2"], partial_data))
    print('confidence intervals:', computeConfidenceIntervals("Y2", ["X1","X2"], partial_data, 'estimateProbability'))
    print()
    print('P(X1 | X2):', shadowRecoveryDim2("Y1", "X1", "R1", "X2", partial_data_R2))
    print('confidence intervals:', computeConfidenceIntervals("Y1", "X1", partial_data_R2, 'shadowRecoveryDim2', RX="R1", C="X2"))
    print()
    print('P(X2)', shadowRecovery("Y1", "X2", "R2", full_data))
    print('confidence intervals:', computeConfidenceIntervals("Y1", "X2", full_data, 'shadowRecovery', RX="R2"))

if __name__ == "__main__":
    np.random.seed(11)

    testShadowYGraph(verbose=False)