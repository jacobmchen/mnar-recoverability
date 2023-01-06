import pandas as pd
import numpy as np
from scipy import optimize
from scipy.special import expit
from mnar_recoverability import estimateProbability
from mnar_ipw_recovery import shadowIpwFun
from shadow_ygraph_fixing import findPropensityScore
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from adjustment import backdoor_adjustment_binary
from adjustment import compute_confidence_intervals

def generateData(size=5000, verbose=False):
    """
    Generate a dataset corresponding to a general MNAR partial Y structure graph.

    Return a 4-tuple in the form (full_data, obs_data, partial_data, weighted_partial_data).

    full_data is the full dataset with no missingness.
    obs_data is the full dataset where missing values of A and Y are replaced with -1.
    partial_data is the dataset where missing values of A and Y are dropped.
    weighted_partial_data is the dataset where partial_data is reweighted according to the
    propensity scores p(R_A | A) and p(R_Y | Y, A, R_A).
    """
    if verbose:
        print("size:", size)

    W1 = np.random.binomial(1, 0.62, size)

    W2 = np.random.binomial(1, 0.54, size)

    A = np.random.binomial(1, expit(W1+W2-0.9), size)
    if verbose:
        print('proportion of A=1', np.bincount(A)[1]/size)

    RA = np.random.binomial(1, expit(A*2), size)
    if verbose:
        print('proportion of RA=1', np.bincount(RA)[1]/size)

    Y = np.random.binomial(1, expit(A*2+W1-1.4), size)
    if verbose:
        print('proportion of Y=1', np.bincount(Y)[1]/size)

    RY = np.random.binomial(1, expit(RA*1.2+Y*1.5+A*1.5-1.2), size)
    if verbose:
        print('proportion of RY=1', np.bincount(RY)[1]/size)
        print()

    # create the fully observed dataset
    full_data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "RA": RA, "RY": RY})

    # create the observed dataset
    Astar = full_data["A"].copy()
    Astar[full_data["RA"] == 0] = -1
    Ystar = full_data["Y"].copy()
    Ystar[full_data["RY"] == 0] = -1

    obs_data = pd.DataFrame({"A": Astar, "Y": Ystar, "W1": W1, "W2": W2, "RA": RA, "RY": RY})

    # create the partially observed dataset
    partial_data = obs_data.copy()
    partial_data = partial_data[partial_data["A"] != -1]
    partial_data = partial_data[partial_data["Y"] != -1]

    # calculate the propensity score p(R_A | A)
    roots_RA = optimize.root(shadowIpwFun, [0.0, 1.0], args=("W1", "A", "RA", obs_data), method='hybr')
    propensity_score_RA = findPropensityScore(roots_RA.x[0], roots_RA.x[1])
    if verbose:
        print("verify propensity scores")
        print("full data", estimateProbability("RA", ["A"], full_data))
        print("shadow recovery p(RA=1 | A=0)", propensity_score_RA[0])
        print("shadow recovery p(RA=1 | A=1)", propensity_score_RA[1])
        print()

    # calculate the propensity score p(R_Y | Y, A, R_A=1)
    # drop rows of data where A is missing
    obs_data_RA1 = obs_data[obs_data["RA"] == 1]

    # create datasets where A=1 and A=0
    obs_data_RA1_A1 = obs_data_RA1[obs_data_RA1["A"] == 1]
    obs_data_RA1_A0 = obs_data_RA1[obs_data_RA1["A"] == 0]

    # recover the propensity score for p(R_Y | Y, A, R_A=1) using the shadow IPW method twice, once where A=1 and
    # once where A=0
    roots_RY_RA1_A1 = optimize.root(shadowIpwFun, [0.0, 1.0], args=("W1", "Y", "RY", obs_data_RA1_A1), method='hybr')
    roots_RY_RA1_A0 = optimize.root(shadowIpwFun, [0.0, 1.0], args=("W1", "Y", "RY", obs_data_RA1_A0), method='hybr')
    propensity_score_RY_RA1_A1 = findPropensityScore(roots_RY_RA1_A1.x[0], roots_RY_RA1_A1.x[1])
    propensity_score_RY_RA1_A0 = findPropensityScore(roots_RY_RA1_A0.x[0], roots_RY_RA1_A0.x[1])

    # create a 4-tuple representing the propensity score p(RY | Y, A, RA=1) in the form
    # (p(RY=1 | Y=0, A=0, RA=1), p(RY=1 | Y=0, A=1, RA=1), p(RY=1 | Y=1, A=0, RA=1), p(RY=1 | Y=1, A=1, RA=1))
    propensity_score_RY = (propensity_score_RY_RA1_A0[0], propensity_score_RY_RA1_A1[0], propensity_score_RY_RA1_A0[1], propensity_score_RY_RA1_A1[1])
    if verbose:
        print("verify propensity scores")
        print("full data", estimateProbability("RY", ["Y", "A", "RA"], full_data))
        print("shadow recovery p(RY=1 | Y=0, A=0, RA=1)", propensity_score_RY[0])
        print("shadow recovery p(RY=1 | Y=0, A=1, RA=1)", propensity_score_RY[1])
        print("shadow recovery p(RY=1 | Y=1, A=0, RA=1)", propensity_score_RY[2])
        print("shadow recovery p(RY=1 | Y=1, A=1, RA=1)", propensity_score_RY[3])
        print()

    # calculate the propensity score for each row of data
    scores_RA = propensity_score_RA[0]**(1-partial_data["A"]) + propensity_score_RA[1]**(partial_data["A"]) - 1
    scores_RY = ( propensity_score_RY[0]**( (1-partial_data["Y"])*(1-partial_data["A"]) ) +
                  propensity_score_RY[1]**( (1-partial_data["Y"])*(partial_data["A"]) ) +
                  propensity_score_RY[2]**( (partial_data["Y"])*(1-partial_data["A"]) ) +
                  propensity_score_RY[3]**( (partial_data["Y"])*(partial_data["A"]) ) - 3)
    
    # create the weights according to the propensity scores
    partial_data["weights"] = 1 / (scores_RA*scores_RY)

    # resample the dataset using the weights, this is the fixing in parallel operation
    weighted_partial_data = partial_data.sample(n=len(partial_data), replace=True, weights="weights")

    return (full_data, obs_data, partial_data, weighted_partial_data)

if __name__ == "__main__":
    np.random.seed(11)

    full_data, obs_data, partial_data, weighted_partial_data = generateData(size=60000, verbose=False)

    # get rid of columns of data that we don't need for causal discovery
    weighted_partial_data.pop("RA")
    weighted_partial_data.pop("RY")
    weighted_partial_data.pop("weights")

    # get rid of columns of data that we don't need for causal discovery
    partial_data.pop("RA")
    partial_data.pop("RY")
    partial_data.pop("weights")

    # require that the shadow variable W1 is a parent of self-censoring variables A and Y
    # X1 -> A, X2 -> Y, X3 -> W1, X4 -> W2
    nodeW1 = GraphNode('X3')
    nodeA = GraphNode('X1')
    nodeY = GraphNode('X2')

    bk = BackgroundKnowledge()
    bk.add_required_by_node(nodeW1, nodeA)
    bk.add_required_by_node(nodeW1, nodeY)

    # run the FCI causal discovery algorithm
    print("running FCI algorithm using reweighted dataset")
    G, edges = fci(weighted_partial_data.to_numpy(), background_knowledge=bk)
    print(G)

    print("running FCI algorithm using partial dataset")
    G, edges = fci(partial_data.to_numpy(), background_knowledge=bk)
    print(G)

    # run the GES causal discovery algorithm
    # print("running GES algorithm using reweighted dataset")
    # record = ges(weighted_partial_data.to_numpy())
    # print(record['G'])

    # print("running GES algorithm using partial dataset")
    # record = ges(partial_data.to_numpy())
    # print(record['G'])

    # estimate the causal effect
    print("estimate causal effect of A on Y")
    print("full dataset")
    print(backdoor_adjustment_binary("Y", "A", ["W1"], full_data))

    print("weighted dataset")
    print(backdoor_adjustment_binary("Y", "A", ["W1"], weighted_partial_data))
    # print(compute_confidence_intervals("Y", "A", ["W1"], weighted_partial_data, "backdoor_binary"))

    print("partial dataset")
    print(backdoor_adjustment_binary("Y", "A", ["W1"], partial_data))
    # print(compute_confidence_intervals("Y", "A", ["W1"], partial_data, "backdoor_binary"))

