import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import optimize
from adjustment import *

class ShadowRecovery:
    """
    class for recovering the causal effect between self-censoring treatment
    and outcome
    """

    def __init__(self, A, Y, R_Y, Z, S, dataset):
        """
        Constructor for the class.
        Initial guesses for the parameters are all 0 except for the coefficient
        of the odds ratio, which starts at 1.
        A, Y, and R_Y should be strings representing the names of the
        treatment, outcome, and missingness
        indicator for the outcome, respectively, as represented in the pandas 
        dataframe dataset.
        Z should be a list of strings representing the names of the variables in the
        adjustment set as represented in dataset.
        S is a string representing the name of the shadow variable as represented in
        dataset.
        dataset should be a dataset already with missing values.
        """
        self.A = A
        self.Y = Y
        self.R_Y = R_Y
        self.Z = Z
        self.S = S
        self.dataset = dataset
        # drop all rows of data where R_Y=0
        self.subset_data = self.dataset[self.dataset[R_Y] == 1]
        # placeholder dataframe to update later
        self.reweight_data = pd.DataFrame()

        # initialize list for the parameters of outcome
        self.paramsY = []

    def _shadowIPWFunctional(self, params, X, R_X):
        """
        Define the shadow IPW functional for this instance of shadow recovery.
        There are two cases: when Z is empty and when Z is non-empty.
        X is the self-censoring variable, and R_X is its missingness indicator
        """
        # outputs represent the outputs of this functional
        outputs = []
        
        # p(R_X = 1 | X = 0)
        # the expit is of the sum of all the parameters multiplied by one parameter
        # when there are no parameters, pRX_X_0 = expit(0)
        pRX_X_0 = expit( np.sum(params[i]*self.dataset[self.Z[i]] for i in range(len(self.Z))) )
        pRX = pRX_X_0 / ( pRX_X_0 + np.exp(params[len(self.Z)]*self.dataset[X])*(1-pRX_X_0) )

        # first k outputs are each parameter individually
        for i in range(len(self.Z)):
            outputs.append( np.average( (self.dataset[self.Z[i]]*self.dataset[R_X]) / pRX - (self.dataset[self.Z[i]]) ) )

        # final output is the average of the shadow variable
        outputs.append( np.average( (np.average(self.dataset[self.S])*self.dataset[R_X]) / pRX - (np.average(self.dataset[self.S])) ) )

        return outputs

    def _initializeParametersGuess(self):
        """
        Initialize a guess for the parameters in shadow recovery. The guesses are 0.0 for all the
        alpha parameters and 1.0 for the gamma parameter.
        """
        guess = []
        
        # when Z has k variables, there should be k+1 parameters
        # draw a number from the normal distribution between 0 and 1 as the initial guess
        for i in range(len(self.Z)):
            guess.append(np.random.uniform(0, 1, 1))
        guess.append(np.random.uniform(0, 1, 1))

        return guess

    def _findRoots(self):
        """
        Estimate the roots for the treatment and outcome.
        """
        guess = self._initializeParametersGuess()
        self.paramsY = optimize.root(self._shadowIPWFunctional, guess, args=(self.Y, self.R_Y), method="hybr")
        print(self.paramsY.success)
        self.paramsY = self.paramsY.x

    def _predictPropensityScores(self):
        """
        Predict the propensity scores for the missingness of the treatment and outcome.
        Predicting the propensity scores depend on the initial functionals used to estimate them.
        """
        # p(R_Y = 1 | Y = 0)
        pRY_Y_0 = expit( np.sum(self.paramsY[i]*self.subset_data[self.Z[i]] for i in range(len(self.Z))) )
        predictionsRY = pRY_Y_0 / (pRY_Y_0 + np.exp(self.paramsY[len(self.Z)]*self.subset_data[self.Y])*(1-pRY_Y_0))

        return predictionsRY
    
    def _reweightData(self):
        propensityScore_RY = self._predictPropensityScores()

        self.subset_data["weights"] = 1/(propensityScore_RY)

        self.reweight_data = self.subset_data.sample(n=len(self.subset_data), replace=True, weights="weights")

    def estimateCausalEffect(self):
        self._findRoots()
        self._reweightData()

        print(self.paramsY)

        return backdoor_adjustment(self.Y, self.A, self.Z, self.reweight_data)

    def confidenceIntervals(self):
        return compute_confidence_intervals(self.Y, self.A, self.Z, self.reweight_data, "backdoor")
