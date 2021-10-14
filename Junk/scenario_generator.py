import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares


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
                               np.sum(np.power(returns-mean,2), axis=0) / returns.shape[0],
                               np.sum(np.power(returns - mean,3), axis=0) / returns.shape[0],
                               np.sum(np.power(returns - mean,3), axis=0) / returns.shape[0]
                               ))
    return moments, corr_matrix


def get_number_of_stocks(TARMOM):
    return np.shape(TARMOM)[0]


def get_MOM(TARMOM):
    MOM = np.column_stack((np.zeros(N_STOCKS),
                           np.ones(N_STOCKS),
                           np.divide(TARMOM[:, 2], np.power(TARMOM[:, 1], 1.5)),
                           np.divide(TARMOM[:, 3], np.power(TARMOM[:, 1], 2))))
    return MOM


def get_TRSFMOM_and_L(MOM, R):
    L = np.linalg.cholesky(R)
    L_bottom = L - np.diagflat(np.diag(L))
    TRSFMOM_1 = np.zeros(N_STOCKS)
    TRSFMOM_2 = np.ones(N_STOCKS)
    TRSFMOM_3 = np.multiply(np.divide(1, np.power(np.diag(L), 3)),
                            MOM[:, 2] - np.matmul(np.power(L_bottom, 3), MOM[:, 2]))
    TRSFMOM_4 = np.multiply(np.divide(1, np.power(np.diag(L), 4)),
                            MOM[:, 3] - 3 - np.matmul(np.power(L_bottom, 4), MOM[:, 3] - 3)) + 3
    return np.column_stack((TRSFMOM_1, TRSFMOM_2, TRSFMOM_3, TRSFMOM_4)), L


def get_moments_of_samples(sampled_outcomes, N_STOCKS,N_OUTCOMES):
    calculated_moments = np.ones(N_STOCKS)
    for i in np.arange(1, 13):
        ith_moment = np.sum(np.power(sampled_outcomes, i), axis=1) / N_OUTCOMES
        calculated_moments = np.column_stack((calculated_moments, ith_moment))
    return calculated_moments


def cubic_transformation(TRSFMOM, calculated_moments, N_STOCKS, N_OUTCOMES):
    def loss_function(x, TRSFMOM, calculated_moments):
        a, b, c, d = x[0], x[1], x[2], x[3]
        coef_formula = np.array([[a, b, c, d, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [a ** 2,
                                  2 * a * b,
                                  2 * a * c + b ** 2,
                                  2 * a * d + 2 * b * c,
                                  2 * b * d + c ** 2,
                                  2 * c * d,
                                  d ** 2,
                                  0, 0, 0, 0, 0, 0],
                                 [a ** 3,
                                  3 * (a ** 2) * b,
                                  3 * (a ** 2) * c + 3 * a * b ** 2,
                                  3 * (a ** 2) * d + 6 * a * b * c + b ** 3,
                                  a * (6 * b * d + 3 * c ** 2) + 3 * (b ** 2) * c,
                                  6 * a * c * d + 3 * (b ** 2) * d + 3 * b * c ** 2,
                                  3 * a * d ** 2 + 6 * b * c * d + c ** 3,
                                  3 * b * d ** 2 + 3 * (c ** 2) * d,
                                  3 * c * d ** 2,
                                  d ** 3,
                                  0, 0, 0],
                                 [a ** 4,
                                  4 * (a ** 3) * b,
                                  4 * (a ** 3) * c + 6 * (a ** 2) * (b ** 2),
                                  4 * (a ** 3) * d + 12 * (a ** 2) * b * c + 4 * a * b ** 3,
                                  (a ** 2) * (12 * b * d + 6 * c ** 2) + 12 * a * (b ** 2) * c + b ** 4,
                                  12 * (a ** 2) * c * d + a * (12 * (b ** 2) * d + 12 * b * c ** 2) + 4 * (b ** 3) * c,
                                  6 * (a ** 2) * d ** 2 + a * (24 * b * c * d + 4 * c ** 3) + 4 * (b ** 3) * d + 6 * (
                                          b ** 2) * c ** 2,
                                  a * (12 * b * d ** 2 + 12 * (c ** 2) * d) + 12 * (b ** 2) * c * d + 4 * b * c ** 3,
                                  12 * a * c * d ** 2 + 6 * (b ** 2) * d ** 2 + 12 * b * (c ** 2) * d + c ** 4,
                                  4 * a * d ** 3 + 12 * b * c * d ** 2 + 4 * (c ** 3) * d,
                                  4 * b * d ** 3 + 6 * (c ** 2) * d ** 2,
                                  4 * c * d ** 3,
                                  d ** 4]])
        loss = np.matmul(coef_formula, calculated_moments) - TRSFMOM
        return loss

    X = []
    for i in np.arange(0, N_STOCKS):
        result = least_squares(loss_function, starting_values, args=[TRSFMOM[i, :], calculated_moments[i, :]],
                               method="trf",
                               max_nfev=10000)
        if not result.success:
            print("Failed to find optimal cubic transformation coefficients.")
        coefs = result.x
        X_i = coefs[0] * np.ones(N_OUTCOMES) + coefs[1] * sampled_outcomes[i, :] + coefs[2] * np.power(
            sampled_outcomes[i, :], 2) + coefs[3] * np.power(sampled_outcomes[i, :], 3)
        X.append(X_i)
    return np.vstack(X)


# -----INPUT PARAMETERS-----
file_name = "data01.csv"
N_OUTCOMES = 5
#max number of iterations in least_squares
#starting point in least_squares

# -----INPUT PHASE-----
#   1) get TARMOM and R
TARMOM, R = get_TARMOM_and_R(file_name)
N_STOCKS = get_number_of_stocks(TARMOM)

#   2) find MOM for Y
MOM = get_MOM(TARMOM)

#   3) compute L and find TRSFMOM for X
TRSFMOM, L = get_TRSFMOM_and_L(MOM, R)

# -----OUTPUT PHASE-----
#   4) generate outcome of X_i for i=1,...,n
#       -sample from N(0,1)
#       -use cubic transformation to match the moments
sampled_outcomes = np.random.normal(0, 1, (N_STOCKS, N_OUTCOMES))
calculated_moments = get_moments_of_samples(sampled_outcomes, N_STOCKS, N_OUTCOMES)

starting_values = np.array([0, 1, 0.5, 0.1])
X = cubic_transformation(TRSFMOM,calculated_moments,N_STOCKS,N_OUTCOMES)

#   5) transform categorical X to Y (to match the correlations)
Y = np.matmul(L, X)

#   6) transform categorical Y to Z (to match moments)
Z = np.matmul(np.diag(np.power(TARMOM[:, 1], 0.5)), Y) + np.reshape(TARMOM[:, 0], (N_STOCKS, 1))

