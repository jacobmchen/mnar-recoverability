import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize
from adjustment import *

class ShadowRecovery:
    """
    class for recovering the causal effect between self-censoring treatment
    and outcome
    """

    def __init__(self, A, R_A, Y, R_Y, Z, S, dataset):
        """
        Constructor for the class.
        Initial guesses for the parameters are all 0 except for the coefficient
        of the odds ratio, which starts at 1.
        A, R_A, Y, and R_Y should be strings representing the names of the
        treatment, missingness indicator for the treatment, outcome, and missingness
        indicator for the outcome, respectively, as represented in the pandas 
        dataframe dataset.
        Z should be a list of strings representing the names of the variables in the
        adjustment set as represented in dataset.
        S is a string representing the name of the shadow variable as represented in
        dataset.
        dataset should be a dataset already with missing values.
        """
        self.A = A
        self.R_A = R_A
        self.Y = Y
        self.R_Y = R_Y
        self.Z = Z
        self.S = S
        self.dataset = dataset
        # drop all rows of data where R_A=0 and R_Y=0
        self.subset_data = self.dataset[self.dataset[R_A] == 1]
        self.subset_data = self.subset_data[self.subset_data[R_Y] == 1]
        # placeholder dataframe to update later
        self.reweight_data = pd.DataFrame()

        # initialize guesses for the parameters of treatment
        self.paramsA = []

        # initialize guesses for the parameters of outcome
        self.paramsY = []

    def _shadowIPWFunctional(self, params, X, R_X):
        """
        Define the shadow IPW functional for this instance of shadow recovery.
        There are two cases: when Z is empty and when Z is non-empty.
        X is the self-censoring variable, and R_X is its missingness indicator
        """
        # outputs represent the outputs of this functional
        outputs = []
        
        if len(self.Z) == 0:
            # use the functional equation when there are no covariates
            # p(R_X = 1 | X = 0)
            # represented by just one parameter
            pRX_X_0 = expit(params[0])
            # the odds ratio functional, where the odds ratio is represented by the exponential function
            # of one parameter (assuming homogenous odds ratio) and the self-censoring variable
            pRX = pRX_X_0 / ( pRX_X_0 + np.exp(params[1]*self.dataset[X])*(1-pRX_X_0) )

            # first output is the shadow variable itself
            outputs.append( np.average( (self.dataset[self.S]*self.dataset[R_X]) / pRX - (self.dataset[self.S]) ) )
            # second output is the shadow variable squared
            outputs.append( np.average( (self.dataset[self.S]**2*self.dataset[R_X]) / pRX - (self.dataset[self.S]**2) ) )
        else:
            # use the functional equation when there are covariates
            # p(R_X = 1 | X = 0)
            # the expit is of the sum of all the parameters multiplied by one parameter
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

        if len(self.Z) == 0:
            # when Z is empty, there are two parameters
            guess = [0.0, 1.0]
        else:
            # when Z has k variables, there should be k+1 parameters
            for i in range(len(self.Z)):
                guess.append(0.0)
            guess.append(1.0)

        return guess

    def _findRoots(self):
        """
        Estimate the roots for the treatment and outcome.
        """
        guess = self._initializeParametersGuess()
        self.paramsA = optimize.root(self._shadowIPWFunctional, guess, args=(self.A, self.R_A), method="hybr").x

        guess = self._initializeParametersGuess()
        self.paramsY = optimize.root(self._shadowIPWFunctional, guess, args=(self.Y, self.R_Y), method="hybr").x

    def _predictPropensityScores(self):
        """
        Predict the propensity scores for the missingness of the treatment and outcome.
        Predicting the propensity scores depend on the initial functionals used to estimate them.
        """
        if len(self.Z) == 0:
            # p(R_A = 1 | A = 0)
            pRA_A_0 = expit(self.paramsA[0])
            predictionsRA = pRA_A_0 / (pRA_A_0 + np.exp(self.paramsA[1]*self.subset_data[self.A])*(1-pRA_A_0))

            # p(R_Y = 1 | Y = 0)
            pRY_Y_0 = expit(self.paramsY[0])
            predictionsRY = pRY_Y_0 / (pRY_Y_0 + np.exp(self.paramsY[1]*self.subset_data[self.Y])*(1-pRY_Y_0))

            return predictionsRA, predictionsRY
        else:
            # p(R_A = 1 | A = 0)
            pRA_A_0 = expit( np.sum(self.paramsA[i]*self.subset_data[self.Z[i]] for i in range(len(self.Z))) )
            predictionsRA = pRA_A_0 / (pRA_A_0 + np.exp(self.paramsA[len(self.Z)]*self.subset_data[self.A])*(1-pRA_A_0))

            # p(R_Y = 1 | Y = 0)
            pRY_Y_0 = expit( np.sum(self.paramsY[i]*self.subset_data[self.Z[i]] for i in range(len(self.Z))) )
            predictionsRY = pRY_Y_0 / (pRY_Y_0 + np.exp(self.paramsY[len(self.Z)]*self.subset_data[self.Y])*(1-pRY_Y_0))

            return predictionsRA, predictionsRY
    
    def _reweightData(self):
        propensityScore_RA, propensityScore_RY = self._predictPropensityScores()

        self.subset_data["weights"] = 1/(propensityScore_RA*propensityScore_RY)

        self.reweight_data = self.subset_data.sample(n=len(self.subset_data), replace=True, weights="weights")

    def estimateCausalEffect(self):
        self._findRoots()
        self._reweightData()

        print(self.paramsA, self.paramsY)

        return backdoor_adjustment_binary(self.A, self.Y, self.Z, self.reweight_data)

    def confidenceIntervals(self):
        return compute_confidence_intervals(self.Y, self.A, self.Z, self.reweight_data, "backdoor_binary")
