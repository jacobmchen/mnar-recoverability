import pandas as pd
import numpy as np
from ratio_test import *
import itertools

class ShadowCovariateSelection:
    """
    Class selecting a valid backdoor adjustment set and shadow variable
    under self-censoring outcome or treatment and outcome.
    """

    def __init__(self, A, R_A, Y, R_Y, dataset, alpha=0.05):
        """
        Constructor for the class.
        A, R_A, Y, and R_Y are strings representing the names of the treatment, missingness
        indicator of treatment, outcome, missingness indicator of outcome, respectively in
        the dataframe dataset.
        dataset is a dataframe representing the dataset with missingness.
        alpha is used as the threshold for independence tests.
        """
        self.A = A
        self.R_A = R_A
        self.Y = Y
        self.R_Y = R_Y
        self.dataset = dataset

        # W is a list containing all the names of the pre-treatment variables
        self.W = []
        for col in self.dataset.columns:
            self.W.append(col)
        self.W.remove(self.A)
        self.W.remove(self.R_A)
        self.W.remove(self.Y)
        self.W.remove(self.R_Y)

        self.alpha = alpha

    def test_independence(self, W_i, Z, condition_A_R_A=False):
        """
        Use the weighted likelihood ratio test to test for independence between R_Y and a 
        candidate shadow variable W_i while conditioning on the set Z.
        """
        # the dataset we will use for this independence test depending on if we are conditioning
        # on A and R_A=1
        this_dataset = pd.DataFrame()
        if condition_A_R_A:
            this_dataset = self.dataset[self.dataset[self.R_A] == 1]
            Z.append(self.A)
        else:
            this_dataset = self.dataset

        p_val = weighted_lr_test(this_dataset, self.R_Y, W_i, Z, state_space="binary")
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

        # iterate over all possible shadow variables
        for i in range(len(self.W)):
            shadowVar = self.W[i]
            # remove the shadow variable from the set Z that we are considering
            fullZ = self.W.copy()
            fullZ.remove(shadowVar)
            
            # iterate over all possible subset lengths of fullZ
            for j in range(len(fullZ)+1):
                # make sure we do not exceed the maximum size
                if j > max_size:
                    break

                subsets = self._find_subsets(fullZ, j)
                print(subsets)
                # iterate over all the subsets for the given length
                for k in range(len(subsets)):
                    # perform the conditional independence tests
                    condition1 = self.test_independence(shadowVar, list(subsets[k]))
                    condition2 = self.test_independence(shadowVar, list(subsets[k]), condition_A_R_A=True)

                    if (not condition1) and condition2:
                        # we have found a valid shadow variable and adjustment set
                        return shadowVar, list(subsets[k])

        # failed to find a valid shadow variable and adjustment set
        return None
