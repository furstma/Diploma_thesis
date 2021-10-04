import time
import datetime

import pandas as pd

from data_download_utils import download_dataset


#DEFINE PARAMETERS
tickers = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
period1 = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
period2 = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
filename = "data01"

#DOWNLOAD and SAVE the data
download_dataset(tickers, period1, period2, interval, filename)




from scenario_generator_MM import *



TARMOM, R = get_TARMOM_and_R("data01.csv")
N_STOCKS = get_number_of_stocks(TARMOM)

N_OUTCOMES = 10
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
for N_OUTCOMES in np.array([10]):
    starting_values = np.random.random_sample(size=(N_STOCKS * N_OUTCOMES))*0.04 + 0.98
    start1 = time.time()
    result = minimize(loss_function, starting_values, args=(TARMOM, R, N_OUTCOMES))
    end1 = time.time()
    result_w = minimize(loss_function_diag, starting_values, args=(TARMOM, R,N_OUTCOMES))
    end2 = time.time()
    print(N_OUTCOMES, ": ", result.success, result.fun, end1-start1,",  |  DIAG:", result_w.success, result_w.fun, end2-end1)






# main workflow scheme
import time
import datetime
import tree_generator

tickers = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
first_valid_date = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
last_valid_date = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
random = False

generator = tree_generator.Generator(tickers, first_valid_date, last_valid_date, random)

compact_tree = generator.generate_compact_tree(3, 20)


# TODO:
# 1) identify correct tree structure
# 2) implement tree_generator


# ad 1)
#   - generate_compact_tree() returns tree in a compact format:
#       list(first_stage_scen, second_stage_scen, ... , last_stage_scen)
#       where ith_stage_scen is pandas dataframe with rows "s1" ... "sN" and cols "STOCK1" ... "STOCKN".