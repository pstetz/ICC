#!/usr/bin/python3
"""
Calculates the Intraclass Correlation Coeffient for two cases

This code has been adapted from
https://www.mathworks.com/matlabcentral/fileexchange/22099-intraclass-correlation-coefficient-icc
"""
import scipy.stats

def finv(p, v1, v2):
    return scipy.stats.f.ppf(p, v1, v2)

def fcdf(p, v1, v2):
    return scipy.stats.f.cdf(p, v1, v2)

def icc(df, icc_type="C-k", alpha=0.05, r0=0):
    n, k = df.shape
    std = df.std()

    SStotal = df.values.var() * ((n * k) - 1)
    MSR = df.mean(axis=1).var() * k
    MSW = df.var(axis=1).sum() / n

    MSC = df.mean(axis=0).var() * n
    MSE = (
        SStotal -
        (MSR * (n - 1)) -
        MSC * (k -1)
    ) / ((n - 1) * (k - 1))
    return case_c_k(MSR, MSE, MSC, MSW, alpha, r0, n, k)

def case_c_k(MSR, MSE, MSC, MSW, alpha, r0, n, k):
    ret  = dict()
    var1 = MSR / MSE
    var2 = 1 - (alpha / 2)
    df1  = n - 1
    df2  = (n - 1) * (k - 1)
    F    = var1 * (1 - r0)

    ret["r"] = (MSR - MSE) / MSR
    ret["F"] = F
    ret["df1"] = df1
    ret["df2"] = df2
    ret["p"] = 1 - fcdf(F, df1, df2);

    FL = var1 / finv(var2, df1, df2)
    FU = var1 * finv(var2, df2, df1)

    ret["LB"] = 1 - (1 / FL)
    ret["UB"] = 1 - (1 / FU)
    return ret

def case_c_1(MSR, MSE, MSC, MSW, alpha, r0, n, k):
    ret  = dict()
    var1 = MSR / MSE
    var2 = 1 - (alpha / 2)
    df1  = n - 1
    df2  = (n - 1) * (k - 1)
    F    = (MSR / MSE) * (1 - r0) / (1 + (k - 1) * r0)

    ret["r"] = (MSR - MSE) / (MSR + (k - 1) * MSE)
    ret["F"] = F
    ret["df1"] = df1
    ret["df2"] = df2
    ret["p"] = 1 - fcdf(F, df1, df2);

    FL = var1 / finv(var2, df1, df2)
    FU = var1 * finv(var2, df2, df1)

    ret["LB"] = 1 - (1 / FL)
    ret["UB"] = 1 - (1 / FU)
    return ret
