import time
import datetime

from Model.data_download_utils import download_dataset


#DEFINE PARAMETERS
tickers = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
period1 = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
period2 = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
filename = "data01"

#DOWNLOAD and SAVE the data
download_dataset(tickers, period1, period2, interval, filename)




from Model.scenario_generator_MM import *


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
from Model import tree_generator
from Model.model import Model

stocks = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
first_valid_date = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
last_valid_date = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
random = False

generator = tree_generator.Generator(stocks, first_valid_date, last_valid_date, random)

scenarios = generator.generate_scenarios([10,8,5,5])

model = Model(scenarios, stocks, {"lmbda": 0.3, "alpha": 0.1, "transaction_cost": 0.02})

#model.print_scenarios()

#model.print()

#model.save_lp_file("10x8x6_test")

#model.solve()

#model.print_results_in_tree()








# testing big trees
import time
import datetime
import numpy as np
from Model import tree_generator
from Model.model import Model

stocks = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
first_valid_date = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
last_valid_date = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
random = False

generator = tree_generator.Generator(stocks, first_valid_date, last_valid_date, random)
TARMOM, R = generator.get_TARMOM_and_R()
results = []
for i in range(4):
    scenarios = generator.generate_scenarios([10,10,10,10])
    model = Model(scenarios, stocks, {"lmbda": 0.3, "alpha": 0.2, "transaction_cost": 0.02})
    model.solve()
    print(model.get_first_stage_result())
    results.append(model.get_first_stage_result())

result_array = np.array([list(dict.values()) for dict in results])










# testing scenario generation by only two moments
# main workflow scheme
import time
import datetime
from Model import tree_generator
from Model.model import Model

stocks = sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
first_valid_date = int(time.mktime(datetime.datetime(2011,1,1,1,1).timetuple()))
last_valid_date = int(time.mktime(datetime.datetime(2020,12,31,23,59).timetuple()))
interval = "1wk"
random = False

generator = tree_generator.Generator(stocks, first_valid_date, last_valid_date, random)

branching = 10
use_only_two_moments = True

scenarios = generator.generate_scenarios([10], use_only_two_moments)
import numpy as np
np.mean(scenarios[0], axis = 0)

TARMOM, R = generator.get_TARMOM_and_R()
