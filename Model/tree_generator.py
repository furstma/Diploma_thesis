from pathlib import Path
import pandas as pd
from Model.data_download_utils import download_dataset
from Model.scenario_generator_MM import generate_scenario


def initialize_dataset(tickers, first_valid_date, last_valid_date):
    dataset_name = '_'.join(tickers) + "_" + str(first_valid_date) + "_" +  str(last_valid_date) + ".pkl"
    if not Path("Data/" + dataset_name).is_file():
        download_dataset(tickers, first_valid_date, last_valid_date, "1wk", dataset_name)

    dataset = pd.read_pickle("Data/" + dataset_name)
    return dataset


class Generator:
    def __init__(self, tickers, first_valid_date: int, last_valid_date: int, random: bool):
        self.tickers = tickers
        self.first_valid_date = first_valid_date
        self.last_valid_date = last_valid_date
        self.random = random

        self.dataset = initialize_dataset(tickers, first_valid_date, last_valid_date)

    def generate_scenarios(self, branching: [int]):
        compact_tree = []
        for i in branching:
            result = generate_scenario(self.dataset, i)
            scenario_array = result.x.reshape((i, -1))
            scenario = pd.DataFrame(scenario_array, columns=self.tickers)
            compact_tree.append(scenario)
            print("Branching " + str(i) + " generated.")
        return compact_tree






