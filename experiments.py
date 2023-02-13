import pandas as pd
import numpy as np
import csv
from scipy.special import expit
from shadow_recovery import ShadowRecovery
from shadow_covariate_selection import ShadowCovariateSelection
from adjustment import *

def generateData(size=5000, verbose=False, possible=True):
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
    
    A = np.random.binomial(1, expit(0.52 + 2*W1 + 2*W2 + 2*W3 + 2*W4), size)
    if verbose:
       print("proportion of A=1:", np.bincount(A)[1]/size)

    Y = np.random.binomial(1, expit(3*A + 2*W2 + 2*W3 + 2*W4), size)
    if verbose:
       print("proportion of Y=1:", np.bincount(Y)[1]/size)
    
    I = np.random.normal(0, 2, size)

    params = [1, 0.5, -1.5, 1.5]

    # flag will help us keep track if we need to treat W4 as an unmeasured confounder
    # if we are treating W4 as an unmeasured confounder, then the flag will remain True
    flag = True
    # we generate the data where there is a possible Z
    if possible:
        pRY_Y_0 = expit(params[0]*W2 + params[0]*W3 + params[0]*W4 + params[1]*I)
        flag = False
    # we generate data where there is no possible Z due to the A to R_Y edge
    else:
        # there's a 0.5 probability that we add an edge A -> R_Y and a 0.5 probability that we
        # treat W4 as an unmeasured confounder
        if np.random.uniform() < 0.5:
            pRY_Y_0 = expit(params[0]*W2 + params[0]*W3 + params[0]*W4 + params[1]*I + params[3]*A)
            flag = False
        else:
            pRY_Y_0 = expit(params[0]*W2 + params[0]*W3 + params[0]*W4 + params[1]*I)

    pRY_1 = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[2]*Y) * (1-pRY_Y_0) )
    R_Y = np.random.binomial(1, pRY_1, size)
    if verbose:
        print("proportion of R_Y=1:", np.bincount(R_Y)[1]/size)

    # create fully observed dataset
    if flag:
        # we need to treat W4 as an unmeasured confounder
        full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "I": I, "R_Y": R_Y})
    else:
        # no need to treat W4 as an unmeasured confounder
        full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "I": I, "R_Y": R_Y})

    # create partially observed dataset
    Y = full_data["Y"].copy()
    Y[full_data["R_Y"] == 0] = -1

    if flag:
        # we need to treat W4 as an unmeasured confounder
        partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "I": I, "R_Y": R_Y})
    else:
        # no need to treat W4 as an unmeasured confounder
        partial_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "I": I, "R_Y": R_Y})

    subset_data = partial_data[partial_data["R_Y"] == 1]

    return (full_data, partial_data, subset_data)

def covariateSelectionExperiment(experimentSize=200):
    # use multiple sizes
    sizes = [2500]
    
    # array to store the results
    results = []

    for i in range(len(sizes)):
        size = sizes[i]

        successPossible = 0
        successNotPossible = 0
        for j in range(experimentSize):
            np.random.seed(j)
            full_data, partial_data, subset_data = generateData(size=size)
            covariateSelection = ShadowCovariateSelection("A", "Y", "R_Y", "I", partial_data, alpha=0.1)
            result = covariateSelection.findAdjustmentSet()

            if result != None:
                W, Z = result
                if Z == ["W2", "W3", "W4"]:
                    successPossible += 1

            # for the test that's not possible, there's a 0.5 probability that we add the edge A->R_Y
            # and a 0.5 probability that we drop one of the W
            full_data, partial_data, subset_data = generateData(size=size, possible=False)
            covariateSelection = ShadowCovariateSelection("A", "Y", "R_Y", "I", partial_data, alpha=0.1)
            result = covariateSelection.findAdjustmentSet()

            if result == None:
                successNotPossible += 1
        
        successProbabilityPossible = successPossible / experimentSize
        successProbabilityNotPossible = successNotPossible / experimentSize
        results.append((successProbabilityPossible, successProbabilityNotPossible))

    return results

def estimationExperiment(experimentSize=200):
    # calculate the ground truth causal effect
    np.random.seed(0)
    full_data, partial_data, subset_data = generateData(size=50000)

    groundTruth = backdoor_adjustment_binary("Y", "A", ["W2", "W3", "W4"], full_data)

    sizes = [500, 2500, 5000, 10000]
    results = []

    for i in range(len(sizes)):
        failMissingnessData = []
        failBackdoorSetData = []
        methodData = []
        correctData = []

        for j in range(experimentSize):
            np.random.seed(j)
            full_data, partial_data, subset_data = generateData(size=sizes[i])

            # fail to adjust for missing data but use a correct backdoor adjustment set
            failMissingness = ShadowRecovery("A", "Y", "R_Y", ["W2", "W3", "W4"], subset_data, ignoreMissingness=True)
            failMissingnessData.append(failMissingness.estimateCausalEffect())

            # fail to use a valid backdoor adjustment set but correctly adjust for missing data
            failBackdoorSet = ShadowRecovery("A", "Y", "R_Y", ["W2", "W3", "W4"], partial_data, useWrongBackdoorSet=True)
            failBackdoorSetData.append(failBackdoorSet.estimateCausalEffect())

            # use the set Z from the covariate selection class
            covariateSelection = ShadowCovariateSelection("A", "Y", "R_Y", "I", partial_data, alpha=0.05)
            result = covariateSelection.findAdjustmentSet()
            if result != None:
                W, Z = result
                recovery = ShadowRecovery("A", "Y", "R_Y", Z, partial_data)
                methodData.append(recovery.estimateCausalEffect())
            else:
                methodData.append(None)

            correctRecovery = ShadowRecovery("A", "Y", "R_Y", ["W2", "W3", "W4"], partial_data)
            correctData.append(correctRecovery.estimateCausalEffect())

        results.append((failMissingnessData, failBackdoorSetData, methodData, correctData))

    return (groundTruth, results)


if __name__ == "__main__":
    # generateData(verbose=True)

    print("running experiments for covariate selection...")
    results = covariateSelectionExperiment(experimentSize=200)

    f = open("covariate_selection_results.txt", "w")
    f.write(str(results))
    f.close()

    print("running experiments for recovery of causal effect...")
    groundTruth, results = estimationExperiment(experimentSize=200)

    f = open("groundTruth.txt", "w")
    f.write(str(groundTruth))
    f.close()

    for i in range(0, 4):
        dataSize = pd.DataFrame({"No missing\nadjustment": results[i][0], "Wrong\nbackdoor set": results[i][1],
                                 "Method": results[i][2], "Correct\nadjustment": results[i][3]})
        dataSize.to_csv("dataSize"+str(i)+".csv", index=False)

    print("experiments finished!")
    