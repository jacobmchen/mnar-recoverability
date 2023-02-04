import pandas as pd
import numpy as np
from ratio_test import *
import itertools

class ShadowCovariateSelection:
    """
    Class selecting a valid backdoor adjustment set and shadow variable
    under self-censoring outcome or treatment and outcome.
    """

    def __init__(self, A, Y, R_Y, I, dataset, alpha=0.05):
        """
        Constructor for the class.
        A, Y, and R_Y are strings representing the names of the treatment, outcome, 
        missingness indicator of outcome, respectively in the dataframe dataset.
        I is a variable representing an incentive for response
        dataset is a dataframe representing the dataset with missingness.
        alpha is used as the threshold for independence tests.
        """
        self.A = A
        self.Y = Y
        self.R_Y = R_Y
        self.I = I
        self.dataset = dataset

        # drop all rows of data where R_Y=0
        self.subset_data = self.dataset[self.dataset[R_Y] == 1]

        # W is a list containing all the names of the pre-treatment variables
        self.W = []
        for col in self.dataset.columns:
            self.W.append(col)
        # remove non-pre-treatment variables from the list
        self.W.remove(self.A)
        self.W.remove(self.Y)
        self.W.remove(self.R_Y)
        self.W.remove(self.I)

        self.alpha = alpha

    def test_independence(self, X, Y, Z, data):
        """
        Use the weighted likelihood ratio test to test for independence between X and Y 
        while conditioning on the set Z. X should be a binary variable.
        """
        # all tests involved will involve a binary variable, so we can use the binary state space
        # print(X, Y, Z)
        p_val = weighted_lr_test(data, X, Y, Z, state_space="binary")
        # print(p_val)
        if p_val < self.alpha:
            return False
        else:
            return True

    def _find_subsets(self, Z, n):
        """
        find all subsets of size n in the list of covariates Z
        """
        return list(itertools.combinations(Z, n))
    
    def findAdjustmentSet(self, max_size=None):
        # if user does not specify a max size, then the max size is simply the 
        # size of the entire set of candidates
        if max_size == None:
            max_size = len(self.W)

        # iterate over all possible S
        for i in range(len(self.W)):
            W = self.W[i]
            # remove the shadow variable from the set Z that we are considering
            fullZ = self.W.copy()
            fullZ.remove(W)
            
            # iterate over all possible subset lengths of fullZ
            for j in range(len(fullZ)+1):
                # make sure we do not exceed the maximum size
                if j > max_size:
                    break

                subsets = self._find_subsets(fullZ, j)
                # iterate over all the subsets for the given length
                for k in range(len(subsets)):
                    # perform the conditional independence tests

                    # check for dependence between missingness of outcome and incentive
                    condition1 = self.test_independence(self.R_Y, self.I, [], self.dataset)

                    # check for independence between incentive and treatment conditional on the subset
                    # this test requires using only data where R_Y=1
                    condition2 = self.test_independence(self.A, self.I, list(subsets[k])+[self.Y], self.subset_data)

                    # check for dependence between S and R_Y conditional on the subset
                    condition3 = self.test_independence(self.R_Y, W, list(subsets[k]), self.dataset)

                    # check for independence between S and R_Y conditional on the subset and the treatment
                    condition4 = self.test_independence(self.R_Y, W, list(subsets[k])+[self.A], self.dataset)

                    if (not condition1) and condition2 and (not condition3) and condition4:
                        # we have found a valid subset that satisfies the four conditions
                        return W, list(subsets[k])

        # failed to find a valid subset that satisfies the four conditions
        return None
