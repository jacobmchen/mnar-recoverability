import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize
from shadow_recovery import ShadowRecovery

def generateData(size=5000, verbose=False):
    """
    Generate a dataset.

    return a 2-tuple: first is a pandas dataframe of the fully observed data,
    second is a pandas dataframe of the partially observed data
    """
    if verbose:
        print(size)

    #W1 = np.random.uniform(0, 1, size)
    W1 = np.random.normal(0, 1, size)
    #W1 = np.random.normal(0, 1, size)*np.random.normal(0, 1, size)
    #W1 = np.random.normal(0, 1, size) + np.random.binomial(1, 0.5, size)
    #W1 = np.random.normal(0, 1, size) + np.random.uniform(0, 1, size)
    #W1 = np.random.multinomial(1, [0.2, 0.3, 0.35, 0.15], size)
    #W1 = np.array([list(w).index(1) for w in W1])

    W3 = np.random.normal(0, 1, size)

    W2 = np.random.normal(0, 1, size) + 2*W1 + 1.5*W3

    W4 = np.random.uniform(0, 1, size) + 2*W1
    
    A = np.random.binomial(1, expit(2*W1+W2), size)
    #if verbose:
    #    print("proportion of A=1:", np.bincount(A)[1]/size)

    #Y = np.random.binomial(1, expit(2*W2 + A), size)
    Y = np.random.normal(0, 1, size) + 0.3*W2 + 2.5*A

    #if verbose:
    #    print("proportion of Y=1:", np.bincount(Y)[1]/size)

    # R_Y=1 denotes observed value of Y
    #params = [0.2, 0.3, -0.6, -1]
    params = [-0.5, 1.2, 0.3]
    #pRY_Y_0 = expit(params[0]*W2 + params[1]*W3)
    pRY_Y_0 = expit(params[1]*W4)
    #pRY_Y_0 = expit(0)
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1
    partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "R_Y": R_Y})

    return (full_data, partial_data)

def shadowIpwFunctional_Y(params, W_1, W_2, W_3, W_4, R_Y, Y, data):
    """
    Define the system of shadow IPW functions for a shadow variable W_1, a covariate W_2, a 
    covariate W_3, missingness indicator R_Y, and self-censoring outcome variable Y.

    params is a list of four variables alpha_0, alpha_1, alpha_2, and gamma_0 in that order

    return values of the functionals based on the parameters
    """
    # p(R_Y=1 | Y=0)
    pRY_Y_0 = expit(params[0]*data[W_2] + params[1]*data[W_3] + params[2]*data[W_4])
    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[3]*data[Y]) * (1-pRY_Y_0) )

    output1 = np.mean( data[W_2]*data[R_Y] / pRY_1 - data[W_2] )
    output2 = np.mean( data[W_3]*data[R_Y] / pRY_1 - data[W_3] )
    output3 = np.mean( data[W_4]*data[R_Y] / pRY_1 - data[W_4] )
    output4 = np.mean( np.mean(data[W_1])*data[R_Y] / pRY_1 - np.mean(data[W_1]) )
    #output4 = np.mean( data[R_Y]/pRY_1 - 1 )

    return [output1, output2, output3, output4]


if __name__ == "__main__":
    #np.random.seed(0)

    full_data, partial_data = generateData(size=50000, verbose=True)
    print()

    S = "W1"
    Z = ["W2", "W3", "W4"]
    shadowRecovery = ShadowRecovery("A", "Y", "R_Y", Z, S, partial_data)
    print(shadowRecovery.estimateCausalEffect())

    ##########################
    # print("find roots of p(R_Y | Y, W2, W3)")
    # roots_RY = optimize.root(shadowIpwFunctional_Y, np.random.uniform(-0.5, 0.5, 4), args=("W1", "W2", "W3", "W4", "R_Y", "Y", partial_data), method='hybr')
    # #roots_RY = optimize.root(shadowIpwFunctional_Y, [0, 0, 0.5, -0.5, 0.5], args=("W1", "W2", "W3", "W4", "R_Y", "Y", partial_data), method='hybr')
    # #roots_RY = optimize.root(shadowIpwFunctional_Y, [0.2, 0.3, -0.6, -1], args=("W1", "W2", "W3", "R_Y", "Y", partial_data), method='hybr')
    # print("converge?", roots_RY.success)
    # print("roots:", roots_RY.x)
    # print()

    # # print the output of the functional using the roots as input
    # print(shadowIpwFunctional_Y(roots_RY.x, "W1", "W2", "W3", "W4", "R_Y", "Y", partial_data))

    # subset_data = partial_data[partial_data["R_Y"] == 1]
    # model = sm.GLM.from_formula(formula="Y ~ W1 + W2 + W3 + W4", data=subset_data, family=sm.families.Gaussian()).fit()
    # print(model.params)
