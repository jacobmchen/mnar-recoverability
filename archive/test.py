import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm

size = 10000
dim = 4
meanVector = [0]*dim
# covariance matrix for the errors
# omega = np.eye(dim)
omega = np.array([[1.2, 0, 0, 0],
                    [0, 1, 0.4, 0.4],
                    [0, 0.4, 1, 0.3],
                    [0, 0.4, 0.3, 1]])

# omega = np.array([[1.2, 0, 0, 0],
#                     [0, 1, 0, 0],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1]])

W = np.random.multivariate_normal(meanVector, omega, size=size)

W1 = W[:,0]
W2 = W[:,1]
W3 = W[:,2]
W4 = W[:,3]

# U1 = np.random.normal(0, 1, size)
# W1 = np.random.normal(0, 1, size)
# W2 = U1 + np.random.normal(0, 1, size)
# W3 = U1 + np.random.normal(0, 1, size)
# W4 = U1 + np.random.normal(0, 1, size)

A = np.random.binomial(1, expit(0.52 + 1*W1 - 1*W2 - 1*W3 + 1*W4), size)
# Y = np.random.binomial(1, expit(3*A + 2*W2 + 2*W3 + 2*W4), size)
Y = 3*A + 2*W2 + 2*W3 + 2*W4 + np.random.normal(0, 1, size)

data = pd.DataFrame({"A": A, "Y": Y, "W1": W1, "W2": W2, "W3": W3, "W4": W4})




model = sm.GLM.from_formula(formula="A~W2+W3+W4", data=data, family=sm.families.Binomial()).fit()
pA = model.predict(data)
print(pA)
print(model.summary())
ipw_est = np.mean(data["Y"]*data["A"]/pA - data["Y"]*(1-data["A"])/(1-pA))
# modelY = sm.GLM.from_formula(formula="Y~A+W2+W3+W4", data=data, family=sm.families.Binomial()).fit()
modelY = sm.GLM.from_formula(formula="Y~A+W2+W3+W4", data=data, family=sm.families.Gaussian()).fit()
print(modelY.summary())
dataA0 = data.copy()
dataA1 = data.copy()
dataA0["A"] = 0
dataA1["A"] = 1
backdoor_est = np.mean(modelY.predict(dataA1) - modelY.predict(dataA0))


estimatesIPW = []
estimatesBackdoor = []

for i in range(500):
    
    data_sample = data.sample(replace=True, n=len(data))
    model = sm.GLM.from_formula(formula="A~W2+W3+W4", data=data_sample, family=sm.families.Binomial()).fit()
    pA = model.predict(data_sample)
    estimatesIPW.append(np.mean(data_sample["Y"]*data_sample["A"]/pA - data_sample["Y"]*(1-data_sample["A"])/(1-pA)))

    modelY = sm.GLM.from_formula(formula="Y~A+W2+W3+W4", data=data_sample, family=sm.families.Gaussian()).fit()
    dataA0 = data_sample.copy()
    dataA1 = data_sample.copy()
    dataA0["A"] = 0
    dataA1["A"] = 1
    estimatesBackdoor.append(np.mean(modelY.predict(dataA1) - modelY.predict(dataA0)))

q = np.quantile(estimatesIPW, q=[0.025, 0.975])
print(ipw_est, q[0], q[1])

q = np.quantile(estimatesBackdoor, q=[0.025, 0.975])
print(backdoor_est, q[0], q[1])
