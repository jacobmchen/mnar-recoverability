import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize

def generateData(size=5000, verbose=False):
    """
    Generate a dataset.

    return a 2-tuple: first is a pandas dataframe of the fully observed data,
    second is a pandas dataframe of the partially observed data
    """
    if verbose:
        print(size)

    W1 = np.random.normal(0, 1, size)

    W3 = np.random.normal(0, 1, size)

    W2 = np.random.normal(0, 1, size) + 2*W1 + 1.5*W3
    
    A = np.random.binomial(1, expit(W1+W2), size)
    if verbose:
        print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.binomial(1, expit(W2+A-0.4), size)
    if verbose:
        print("proportion of Y=1:", np.bincount(Y)[1]/size)

    # R_A=1 denotes observed value of A
    params = [0.33, 0.9]
    pRA_A_0 = expit(params[0])
    pRA_1 = pRA_A_0 / ( pRA_A_0 + np.exp(params[1]*A) * (1-pRA_A_0) )
    R_A = np.random.binomial(1, pRA_1, size)
    if verbose:
        print("proportion of R_A=1:", np.bincount(R_A)[1]/size)

    # R_Y=1 denotes observed value of Y
    params = [0.4, 0.5, 0.7, 0.9]
    pRY_Y_0 = expit(params[0]+params[1]*W2+params[2]*W3)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[3]*Y) * (1-pRY_Y_0) )
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
    Define the system of shadow IPW functions for a shadow variable W_1, a covariate W_2, a 
    covariate W_3, missingness indicator R_Y, and self-censoring outcome variable Y.

    params is a list of four variables alpha_0, alpha_1, alpha_2, and gamma_0 in that order

    return values of the functionals based on the parameters
    """
    # p(R_Y=1 | Y=0)
    pRY_Y_0 = expit(params[0]+params[1]*data[W_2]+params[2]*data[W_3])
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[3]*data[Y]) * (1-pRY_Y_0) )

    output1 = np.average( ((data[W_1])*data[R_Y]) / pRY_1 - (data[W_1]) )
    output2 = np.average( ((data[W_2])*data[R_Y]) / pRY_1 - (data[W_2]) )
    output3 = np.average( ((data[W_3])*data[R_Y]) / pRY_1 - (data[W_3]) )
    output4 = np.average( ((1)*data[R_Y]) / pRY_1 - (1) )

    return [output1, output2, output3, output4]

def shadowIpwFunctional_A(params, W_1, R_A, A, data):
    """
    Define the system of shadow IPW functions for a shadow variable W_1, a missingness indicator
    R_A, and self-censoring outcome variable A.

    params is a list of two variables alpha_0, gamma_0 in that order

    return values of the functionals based on the parameters
    """
    # p(R_Y=1 | Y=0)
    pRA_A_0 = expit(params[0])
    pRA_1 = pRA_A_0 / (pRA_A_0 + np.exp(params[1]*data[A])*(1-pRA_A_0))

    output1 = np.average( ((data[W_1]*data[R_A])) / pRA_1 - (data[W_1]) )
    output2 = np.average( ((1)*data[R_A]) / pRA_1 - (1) )

    return [output1, output2]

if __name__ == "__main__":
    np.random.seed(10)

    full_data, partial_data = generateData(size=50000, verbose=True)
    print()

    ##########################
    print("find roots of p(R_A | A)")
    # roots_RA = optimize.root(shadowIpwFunctional_A, [0.0, 1.0], args=("W1", "R_A", "A", partial_data), method='hybr')
    roots_RA = optimize.root(shadowIpwFunctional_A, [0.0, 1.0], args=("W1", "R_A", "A", partial_data), method='hybr')
    print("converge?", roots_RA.success)
    print("roots:", roots_RA.x)
    print()

    ##########################
    print("find roots of p(R_Y | Y, W2, W3)")
    roots_RY = optimize.root(shadowIpwFunctional_Y, [0.0, 0.0, 0.0, 1.0], args=("W1", "W2", "W3", "R_Y", "Y", partial_data), method='hybr')
    print("converge?", roots_RY.success)
    print("roots:", roots_RY.x)
    print()

    # print the output of the functional using the roots as input
    # print(shadowIpwFunctional_Y(roots_RY.x, "W1", "W2", "W3", "R_Y", "Y", partial_data))
