import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

def weighted_lr_test(data, Y, Z, cond_set=[], weights=None, state_space="continuous"):
    """
    Perform a weighted likelihood ratio test for a Verma constraint Y _||_ Z | cond_set in a reweighted distribution
    Y: name of outcome variable (assumed to be a continuous variable)
    Z: name of anchor variable
    cond_set: list of variable names that are conditioned on when checking independence
    state_space: "continuous" if Y is a continuous variable, otherwise Y is treated as binary

    This function is taken from Bhattacharya, 2022: https://github.com/rbhatta8/fdt
    """

    # fit weighted null and alternative models to check independence in the kernel
    formula_Y = Y + "~ 1"
    if len(cond_set) > 0:
        formula_Y += " + " + "+".join(cond_set)

    if weights is None:
        weights = np.ones(len(data))

    if state_space == "continuous":
        modelY_null = sm.GLM.from_formula(formula=formula_Y, data=data, freq_weights=weights, family=sm.families.Gaussian()).fit()
        modelY_alt = sm.GLM.from_formula(formula=formula_Y + "+" + Z, data=data, freq_weights=weights, family=sm.families.Gaussian()).fit()
    elif state_space == "binary":
        modelY_null = sm.GLM.from_formula(formula=formula_Y, data=data, freq_weights=weights, family=sm.families.Binomial()).fit()
        modelY_alt = sm.GLM.from_formula(formula=formula_Y + "+" + Z, data=data, freq_weights=weights, family=sm.families.Binomial()).fit()
    else:
        print("Invalid state space for outcome.")
        assert(False)

    # the test statistic 2*(loglike_alt - loglike_null) is chi2 distributed
    chi2_stat = 2*(modelY_alt.llf - modelY_null.llf)
    return 1 - stats.chi2.cdf(x=chi2_stat, df=1)