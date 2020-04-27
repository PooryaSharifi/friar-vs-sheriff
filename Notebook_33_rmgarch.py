import rpy2.robjects as objects
import numpy as np

import random
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# https://quant.stackexchange.com/questions/20687/multivariate-garch-in-python
# pd_rets - a pandas dataframe of daily returns, where the column names are the tickers of stocks and index is the trading days.
pd_rets = pd.DataFrame({'GOOG': [random.random() for _ in range(1200)],
                        'AMZN': [random.random() for _ in range(1200)],
                        'GE': [random.random() for _ in range(1200)]})
n_days = 100

# compute DCC-Garch in R using rmgarch package
pandas2ri.activate()
with localconverter(ro.default_converter + pandas2ri.converter):
    r_rets = ro.conversion.py2rpy(pd_rets)
# r_rets = pandas2ri.py2ri(pd_rets)  # convert the daily returns from pandas dataframe in Python to dataframe in R
r_dccgarch_code = """
                library('rmgarch')
                function(r_rets, n_days){
                        univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                                                    variance.model = list(garchOrder = c(1,1),
                                                                        variance.targeting = FALSE, 
                                                                        model = "sGARCH"),
                                                    distribution.model = "norm")
                        n <- dim(r_rets)[2]
                        dcc_spec <- dccspec(uspec = multispec(replicate(n, univariate_spec)),
                                            dccOrder = c(1,1),
                                            distribution = "mvnorm")
                        dcc_fit <- dccfit(dcc_spec, data=r_rets)
                        forecasts <- dccforecast(dcc_fit, n.ahead = n_days)
                        list(dcc_fit, forecasts@mforecast$H)
                }
                """
r_dccgarch = objects.r(r_dccgarch_code)
r_res = r_dccgarch(r_rets, n_days)
pandas2ri.deactivate()
# end of R

print(r_res)
r_dccgarch_model = r_res[0]  # model parameters
r_forecast_cov = r_res[1]  # forecasted covariance matrices for n_days

# access and transform the covariance matrices in R format
n_cols = pd_rets.shape[1]  # get the number of stocks in pd_rets
n_elements = n_cols*n_cols  # the number of elements in each covariance matrix
n_matrix = int(len(r_forecast_cov[0]) / (n_elements))
print(n_matrix)  # this should be equal to n_days

# sum the daily forecasted covariance matrices
cov_matrix = 0
for i in range(n_matrix):
    i_matrix = np.array([v for v in r_forecast_cov[0][i*n_elements:(i+1)*n_elements]])
    i_matrix = i_matrix.reshape(n_cols,n_cols)
    cov_matrix += i_matrix

print('######')
print(type(cov_matrix))
print(cov_matrix.shape)
print(cov_matrix)
print(cov_matrix[0, 1] / cov_matrix[0, 0] ** .5 / cov_matrix[1, 1] ** .5)
