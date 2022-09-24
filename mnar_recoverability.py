import pandas as pd
import numpy as np
from scipy.special import expit

def estimateProbability(Y, Z, data):
    """
    Given Y, a binary variable, and a list Z of binary variables, calculates the ratio
    P(Y=1 | Z) using data for each possible setting of binary variables in Z. The
    length of the output will be 2**len(Z).
    """
    original_data = data.copy()
    # save the size of the original data set
    size = len(data)

    # initialize the output list
    probability_table = []

    binstrings = 2**len(Z)
    # length of the binary strings will be log(binstrings)
    binstring_length = len(Z)
    # enumerate the possible binary strings from 0 to 2**len(Z)
    for i in range(0, binstrings):
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
        state_string = state_string[0:len(state_string)-2]

        # append to the output
        # divide the amount of rows of data where Y == 1 in the truncated data set by
        # the rows of data in the original data set
        probability_table.append((state_string, len(data[data[Y] == 1])/size))

    return probability_table

def generateShadowGraph():
    """
    Generate a graph with the following format: Y->X->RX. We refer to this graph as the
    shadow graph. Both Y and X are binary variables. RX is the missingness indicator for the
    variable X. RX=1 indicates that the value of X is missing, and RX=0 indicates that the
    value of X is observed.
    """
    size = 5000
    print("size:", size)

    # around 0.62 of rows of data of Y are 1
    Y = np.random.binomial(1, 0.62, size)
    print(np.bincount(Y)[1]/size)
    # around 0.52 of rows of data of X are 1
    X = np.random.binomial(1, expit(Y*0.7-0.4), size)
    print(np.bincount(X)[1]/size)
    # generate the missingness mechanism of X, around 0.29 of the rows of X are
    # missing
    RX = np.random.binomial(1, expit(X*0.5-1.2), size)
    print(np.bincount(RX)[1]/size)
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

    # estimate the conditional probabilities of P(Y | X) and P(X) for the fully observed
    # data set
    print('fully observed data set')
    print('P(Y | X): ', estimateProbability("Y", ["X"], full_data))

    # estimate the conditional probabilities of P(Y | X) and P(X) for the partially observed
    # data set
    print('partially observed data set')
    print('P(Y | X):', estimateProbability("Y", ["X"], partial_data))

if __name__ == "__main__":
    np.random.seed(10)

    generateShadowGraph()