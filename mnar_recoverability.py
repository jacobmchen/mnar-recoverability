import pandas as pd
import numpy as np
from scipy.special import expit

def estimateProbability(Y, Z, data):
    """
    Given Y, a binary variable, and a list Z of binary variables, calculates the ratio
    P(Y=1 | Z) using data for each possible setting of binary variables in Z. The
    length of the output will be 2**len(Z).

    Returns a probability table which is a list of tuples. The first element of the 
    tuple is a string representing the probability law, and the second element is
    a float of the value of the probability law.
    """
    # save a copy of the original data set
    original_data = data.copy()
    # initialize a size variable
    size = 0

    # initialize the output list
    probability_table = []

    # handle edge case where Z is an empty list
    # in this case return P(Y=1)
    if len(Z) == 0:
        # divide by the size of the whole data set
        size = len(data)
        probability_table.append(('P(Y=1)', len(data[data[Y] == 1])/size))
        return probability_table

    binstrings = 2**len(Z)
    # length of the binary strings will be log(binstrings)
    binstring_length = len(Z)
    # enumerate the possible binary strings from 0 to 2**len(Z)
    for i in range(binstrings):
        data = original_data.copy()
        format_string = '0' + str(binstring_length) + 'b'
        this_state = format(i, format_string)

        # create string representing this state
        state_string = 'P(Y=1 | '

        # initialize a counter
        ctr = 0
        for bit in this_state:
            data = data[data[Z[ctr]] == int(bit)]
            state_string += Z[ctr] + '=' + bit + ', '
            ctr += 1

        # the last two characters in state_string are extraneous as they are a comma and a space
        # but we also need to add a parantheses
        state_string = state_string[0:len(state_string)-2] + ')'

        # append to the output
        # divide the amount of rows of data where Y == 1 in the truncated data set by
        # the rows of data in the truncated data set
        size = len(data)
        probability_table.append((state_string, len(data[data[Y] == 1])/size))

    return probability_table

def estimateProbabilityTwoVar(Y, Z, data, verbose=False):
    """
    Given Y, a binary variable and a binary variable Z, calculates the ratio
    P(Y=1 | Z=1) and P(Y=1 | Z=0).

    Returns a probability table which is a list of tuples. The first element of the 
    tuple is a string representing the probability law, and the second element is
    a float of the value of the probability law.
    """
    if verbose:
        print(data)

    # save a copy of the original data set
    original_data = data.copy()
    # initialize a size variable
    size = len(data)
    if verbose:
        print('size of original data set:', size)

    # initialize the output list
    probability_table = []

    # create a copy of the original data set
    data = original_data.copy()
    # first calculate P(Y=1 | X=1)
    # subset the data to rows where Z=1
    data = data[data[Z] == 1]
    size = len(data)
    if verbose:
        print('rows of data where ' + Z + '=1', len(data))
        print('size of subsetted data set:', size)
    # find the rows of data where Y=1 in the subsetted data set
    probability_table.append(('P(Y=1 | ' + Z + '=1', len(data[data[Y] == 1])/size))

    # create a copy of the original data set
    data = original_data.copy()
    # first calculate P(Y=1 | X=1)
    # subset the data to rows where Z=1
    data = data[data[Z] == 0]
    size = len(data)
    if verbose:
        print('rows of data where ' + Z + '=0', len(data))
        print('size of subsetted data set:', size)
    # find the rows of data where Y=1 in the subsetted data set
    probability_table.append(('P(Y=1 | ' + Z + '=0', len(data[data[Y] == 1])/size))

    return probability_table

def computeConfidenceIntervals(Y, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for estimating probabilities using bootstrap.

    Returns a tuple (q_low, q_high) representing the lower and upper quantiles of the confidence
    interval.
    """
    Ql = alpha/2
    Qu = 1 - alpha/2
    # declare a dictionary that will store a mapping between the conditional probability
    # and its estimates
    estimates = {}

    for i in range(num_bootstraps):
        # resample the data with replacement
        bootstrap_data = data.sample(len(data), replace=True)
        bootstrap_data.reset_index(drop=True, inplace=True)

        # add estimate from resampled data
        output = estimateProbability(Y, Z, bootstrap_data)
        for estimate in output:
            # if this conditional probability not in estimates yet, initialize an 
            # empty list
            if estimate[0] not in estimates:
                estimates[estimate[0]] = []
            estimates[estimate[0]].append(estimate[1])

    # iterate over the keys in estimates
    for probabilityLaw in estimates:
        # calculate the quantiles
        quantiles = np.quantile(estimates[probabilityLaw], q=[Ql, Qu])
        q_low = quantiles[0]
        q_up = quantiles[1]

        # update the dictionary to store the quantile
        estimates[probabilityLaw] = (q_low, q_up)

    return estimates

def testShadowGraph(verbose=False):
    """
    Generate a graph with the following format: Y->X->RX. We refer to this graph as the
    shadow graph. Both Y and X are binary variables. RX is the missingness indicator for the
    variable X. RX=1 indicates that the value of X is missing, and RX=0 indicates that the
    value of X is observed.
    Then test the full law of the shadow graph using conditional probabilities.
    """
    size = 5000
    if verbose:
        print("size:", size)

    # around 0.62 of rows of data of Y are 1
    # Y has no parents, so it does not depend on any other variables
    Y = np.random.binomial(1, 0.62, size)
    if verbose:
        print('proportion of Y=1:', np.bincount(Y)[1]/size)

    # around 0.52 of rows of data of X are 1
    # toggle one of the following DGP processes:
    X = np.random.binomial(1, expit(Y*0.7-0.4), size)
    #X = np.random.binomial(1, 0.52, size)
    if verbose:
        print('proportion of X=1:', np.bincount(X)[1]/size)
    
    # generate the missingness mechanism of X, around 0.29 of the rows of X are
    # missing
    # toggle one of the following DGP processes:
    RX = np.random.binomial(1, expit(X-1.7), size)
    #RX = np.random.binomial(1, 0.28, size)
    #RX = np.random.binomial(1, expit(Y-2), size)
    if verbose:
        print('proportion of RX=1', np.bincount(RX)[1]/size)
    # assert that less than 0.3 of the values of X are missing
    assert RX.sum() <= 0.3*size, 'too many missing values in X'

    # create the fully observed data set
    full_data = pd.DataFrame({"Y": Y, "X": X})

    # create the partially observed data set
    Xstar = X.copy()
    # set missing values to -1 to denote a ? whenever RX == 1
    Xstar[RX == 1] = -1
    partial_data = pd.DataFrame({"Y": Y, "X": Xstar})
    # drop the rows of data where X is unobserved
    partial_data = partial_data[partial_data["X"] != -1]

    # estimate the conditional probabilities for the fully observed data set
    print()
    print('fully observed data set')
    print('P(Y):', estimateProbability("Y", [], full_data))
    print('confidence intervals:', computeConfidenceIntervals("Y", [], full_data))
    print('P(Y | X):', estimateProbability("Y", ["X"], full_data))
    print('confidence intervals:', computeConfidenceIntervals("Y", ["X"], full_data))

    # estimate the conditional probabilities for the partially observed data set
    # note that every single probability calculated here has RX=0 past the
    # conditioning bar
    print()
    print('partially observed data set')
    print('P(Y):', estimateProbability("Y", [], partial_data))
    print('confidence intervals:', computeConfidenceIntervals("Y", [], partial_data))
    print('P(Y | X):', estimateProbability("Y", ["X"], partial_data))
    print('confidence intervals:', computeConfidenceIntervals("Y", ["X"], partial_data))

if __name__ == "__main__":
    np.random.seed(10)

    testShadowGraph(verbose=True)