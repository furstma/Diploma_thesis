

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
