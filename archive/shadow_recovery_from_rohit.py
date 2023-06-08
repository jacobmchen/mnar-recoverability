import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import optimize
from adjustment import *
import statsmodels.api as sm

class ShadowRecovery:
    """
    Class for recovering the causal effect under self-censoring outcome.
    """

    def __init__(self, A, Y, R_Y, Z, dataset, ignoreMissingness=False, useWrongBackdoorSet=False):
        """
        Constructor for the class.

        Due to the completeness condition of using shadow variables, the outcome
        cannot have more possible values than the treatment. For this implementation, 
        A is a binary variable; hence, Y should be a binary variable as well.

        A, Y, and R_Y should be strings representing the names of the
        treatment, outcome, and missingness indicator for the outcome, respectively, 
        as represented in the pandas dataframe dataset.

        Z should be a list of strings representing the names of the variables in the
        adjustment set as represented in dataset.

        dataset should be a dataset already with missing values for the outcome where
        missing values of the outcome are represented by -1.
        """
        self.A = A
        self.Y = Y
        self.R_Y = R_Y
        self.Z = Z
        self.dataset = dataset
        self.size = len(self.dataset)

        # drop all rows of data where R_Y=0
        self.subset_data = self.dataset[self.dataset[R_Y] == 1]

        # initialize list for the parameters of outcome
        self.paramsY = self._initializeParametersGuess()

        self.ignoreMissingness = ignoreMissingness
        self.useWrongBackdoorSet = useWrongBackdoorSet

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

    def _estimatingEquations(self, params):
        """
        Define the estimating equations for shadow recovery.
        There are two cases: when Z is empty and when Z is non-empty.
        """
        # outputs represent the outputs of this functional
        outputs = []
        
        # p(R_Y = 1 | Y = 0) = the sum of all the parameters multiplied by each variable in Z
        # then put into an expit
        # when there are no parameters, pRX_X_0 = expit(0)
        pRY_Y_0 = expit( np.sum(params[i]*self.dataset[self.Z[i]] for i in range(len(self.Z))) )
        # the final parameter is used to estmate OR(R_Y=0, Y)
        pRY = pRY_Y_0 / ( pRY_Y_0 + np.exp(params[len(self.Z)]*self.dataset[self.Y])*(1-pRY_Y_0) )

        # first k equations are each variable in Z individually
        for i in range(len(self.Z)):
            outputs.append( np.average( (self.dataset[self.Z[i]]*self.dataset[self.R_Y]) / pRY - (self.dataset[self.Z[i]]) ) )

        # final equation is the average of the shadow variable, in this case the treatment
        outputs.append( np.average( (np.average(self.dataset[self.A])*self.dataset[self.R_Y]) / pRY - (np.average(self.dataset[self.A])) ) )

        return outputs

    def _findRoots(self):
        """
        Estimate the roots for the treatment and outcome.
        """
        self.paramsY = optimize.root(self._estimatingEquations, self.paramsY, method="hybr")
        # print(self.paramsY.success)
        self.paramsY = self.paramsY.x
        print(self.paramsY)

    def _propensityScoresRY(self, data):
        """
        Predict the propensity scores for the missingness of the outcome using the recovered
        parameters.
        """
        # p(R_Y = 1 | Y = 0)
        pRY_Y_0 = expit( np.sum(self.paramsY[i]*data[self.Z[i]] for i in range(len(self.Z))) )
        propensityScoresRY = pRY_Y_0 / (pRY_Y_0 + np.exp(self.paramsY[len(self.Z)]*data[self.Y])*(1-pRY_Y_0))

        return propensityScoresRY

    def _propensityScoresA(self, data):
        """
        Predict the propensity scores for the outcome given the backdoor adjustment set
        Z.
        """
        # if Z is empty, then there is no need to fit a model and we can just return 
        # the marginal distribution of the treatment A
        if len(self.Z) == 0:
            pA1 = np.bincount(self.dataset[self.A])[1]/self.size

            propensityScoresA = 0*data[self.A] + pA1
            
            return propensityScoresA

        # if Z is non-empty, we use a linear regression to predict the propensity score of 
        # the treatment given the adjustment set Z
        formula = self.A + "~"
        if self.useWrongBackdoorSet:
            # if we purposefully use the wrong backdoor set, we can the last element in Z
            formula += "+".join(self.Z[:-1])
        else:
            # otherwise use the correct backdoor set, which is the entire set Z
            formula += "+".join(self.Z)

        # we may fit the model using the full dataset since the model does not depend on
        # any missing values
        model = sm.GLM.from_formula(formula=formula, data=self.dataset, family=sm.families.Binomial()).fit()
        # print(model.summary())

        # make predictions only for the subsetted data
        propensityScoresA = model.predict(data)
        
        return propensityScoresA

    def _clipping(self, propensityScores, low=0.01, high=0.99):
        
        clippedScores = []
        for p in propensityScores:
            if p < low:
                clippedScores.append(low)
            elif p > high:
                clippedScores.append(high)
            else:
                clippedScores.append(p)

        return np.array(clippedScores)

    def _inverseProbabilityWeightsEstimator(self, data):
        # only need to calculate propensity scores for R_Y if we are not ignoring missingness
        if not self.ignoreMissingness:
            self._findRoots()
            # print(self.paramsY)
            propensityScoresRY = self._propensityScoresRY(data)
            propensityScoresRY = self._clipping(propensityScoresRY)

            # print(propensityScoresRY)

        propensityScoresA = self._propensityScoresA(data)
        propensityScoresA = self._clipping(propensityScoresA)
        # print(propensityScoresA)

        # inverse-probability weight estimator where A is used directly as an indicator function
        # if we are ignoring missingness, then we only need the propensity scores for A

        if self.ignoreMissingness:
            Y0 = np.average( (data[self.Y] * (1-data[self.A])) / (1-propensityScoresA) )
        else:
            Y0 = np.average( (data[self.Y] * (1-data[self.A]) * data[self.R_Y]) / (propensityScoresRY*(1-propensityScoresA)) )

        if self.ignoreMissingness:
            Y1 = np.average( (data[self.Y] * (data[self.A])) / (propensityScoresA) )
        else:
            Y1 = np.average( (data[self.Y] * (data[self.A]) * data[self.R_Y]) / (propensityScoresRY*propensityScoresA) )

        return Y1-Y0
        
    def estimateCausalEffect(self):
        return self._inverseProbabilityWeightsEstimator(self.dataset)

    def confidenceIntervals(self, num_bootstraps=200, alpha=0.05):
        
        Ql = alpha/2
        Qu = 1 - alpha/2
        estimates = []
        
        for i in range(num_bootstraps):
            
            # resample the data with replacement
            data_sampled = self.dataset.sample(len(self.dataset), replace=True)
            data_sampled.reset_index(drop=True, inplace=True)
            
            estimates.append(self._inverseProbabilityWeightsEstimator(data_sampled))

        # calculate the quantiles
        quantiles = np.quantile(estimates, q=[Ql, Qu])
        q_low = quantiles[0]
        q_up = quantiles[1]
        
        return q_low, q_up
