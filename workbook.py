



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
