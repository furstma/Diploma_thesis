import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

def get_TARMOM_and_R(file_name):
    data = pd.read_csv('Data\\' + file_name, delimiter=';').iloc[:, 1:]
    returns = data.pct_change() + 1
    returns = returns.iloc[1:, :]
    corr_matrix = returns.corr(method='pearson').to_numpy()
    if not np.all(np.linalg.eigvals(corr_matrix) > 0):
        print("Correlation matrix R is not positive definite!")
    returns = returns.to_numpy()
    # moments = np.column_stack((np.mean(returns, axis=0),
    #                            stats.tvar(returns, axis=0),
    #                            stats.skew(returns, axis=0),
    #                            stats.kurtosis(returns, axis=0)
    #                            ))
    mean = np.sum(returns, axis=0) / returns.shape[0]
    moments = np.column_stack((mean,
                               np.sum(np.power(returns - mean, 2), axis=0) / returns.shape[0],
                               np.sum(np.power(returns - mean, 3), axis=0) / returns.shape[0],
                               np.sum(np.power(returns - mean, 3), axis=0) / returns.shape[0]
                               ))
    return moments, corr_matrix


def get_number_of_stocks(TARMOM):
    return np.shape(TARMOM)[0]


def loss_function(x, TARMOM, R, N_OUTCOMES):
    x= np.reshape(x, (-1,N_OUTCOMES))
    mean = np.mean(x, axis=1)
    variance = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 2), axis=1) / N_OUTCOMES
    third = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 3), axis=1) / N_OUTCOMES
    fourth = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 4), axis=1) / N_OUTCOMES

    R_discrete = np.corrcoef(x)

    loss = np.sum(np.power(mean - TARMOM[:, 0], 2)) + \
           np.sum(np.power(variance - TARMOM[:, 1], 2)) + \
           np.sum(np.power(third - TARMOM[:, 2], 2)) + \
           np.sum(np.power(fourth - TARMOM[:, 3], 2)) + \
           np.sum(np.power(R_discrete - R, 2))

    return loss

def loss_function_diag(x, TARMOM, R, N_OUTCOMES):
    x= np.reshape(x, (-1,N_OUTCOMES))
    mean = np.mean(x, axis=1)
    variance = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 2), axis=1) / N_OUTCOMES
    third = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 3), axis=1) / N_OUTCOMES
    fourth = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 4), axis=1) / N_OUTCOMES

    R_discrete = np.corrcoef(x)

    loss = np.sum(np.power(mean - TARMOM[:, 0], 2)) + \
           np.sum(np.power(variance - TARMOM[:, 1], 2)) + \
           np.sum(np.power(third - TARMOM[:, 2], 2)) + \
           np.sum(np.power(fourth - TARMOM[:, 3], 2)) + \
           np.sum(np.power(np.triu(R_discrete - R,1), 2))

    return loss


def weighted_loss_function(x, TARMOM, R, N_OUTCOMES, weights):
    x= np.reshape(x, (-1,N_OUTCOMES))
    mean = np.mean(x, axis=1)
    variance = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 2), axis=1) / N_OUTCOMES
    third = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 3), axis=1) / N_OUTCOMES
    fourth = np.sum(np.power(x - np.tile(mean, (N_OUTCOMES, 1)).transpose(), 4), axis=1) / N_OUTCOMES

    R_discrete = np.corrcoef(x)

    loss = weights[0]*np.sum(np.power(mean - TARMOM[:, 0], 2)) + \
           weights[1]*np.sum(np.power(variance - TARMOM[:, 1], 2)) + \
           weights[2]*np.sum(np.power(third - TARMOM[:, 2], 2)) + \
           weights[3]*np.sum(np.power(fourth - TARMOM[:, 3], 2)) + \
           weights[4]*np.sum(np.power(R_discrete - R, 2))

    return loss


TARMOM, R = get_TARMOM_and_R("data01.csv")
N_STOCKS = get_number_of_stocks(TARMOM)

N_OUTCOMES = 3
starting_values = np.random.random_sample(size=(N_STOCKS * N_OUTCOMES))*0.04 + 0.98
result = minimize(loss_function, starting_values, args=(TARMOM, R,N_OUTCOMES))

for N_OUTCOMES in np.array([3,5,7,10,15,30]):
    starting_values = np.random.random_sample(size=(N_STOCKS * N_OUTCOMES))*0.04 + 0.98
    result = minimize(loss_function, starting_values, args=(TARMOM, R,N_OUTCOMES))
    print(result.success, result.fun)





weights =np.array([1,1,1,1,0])
print()
print(weights)
for N_OUTCOMES in np.array([3,5,7,10,15,30]):
    starting_values = np.random.random_sample(size=(N_STOCKS * N_OUTCOMES))*0.04 + 0.98
    start1 = time.time()
    result = minimize(loss_function, starting_values, args=(TARMOM, R, N_OUTCOMES))
    end1 = time.time()
    result_w = minimize(weighted_loss_function, starting_values, args=(TARMOM, R,N_OUTCOMES, weights))
    end2 = time.time()
    print(N_OUTCOMES, ": ", result.success, result.fun, end1-start1,",  |  weighted:", result_w.success, result_w.fun, end2-end1)




#comparison of diag and non-diag
print()
for N_OUTCOMES in np.array([50]):
    starting_values = np.random.random_sample(size=(N_STOCKS * N_OUTCOMES))*0.04 + 0.98
    start1 = time.time()
    result = minimize(loss_function, starting_values, args=(TARMOM, R, N_OUTCOMES))
    end1 = time.time()
    result_w = minimize(loss_function_diag, starting_values, args=(TARMOM, R,N_OUTCOMES))
    end2 = time.time()
    print(N_OUTCOMES, ": ", result.success, result.fun, end1-start1,",  |  DIAG:", result_w.success, result_w.fun, end2-end1)