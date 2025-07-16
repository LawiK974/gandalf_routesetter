import os
import numpy as np
import csv
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/problems.json")
HOLDS_EVAL_PATH = os.path.join(SCRIPT_PATH, "../dataset/2019.csv")
ENVERGURE = 170  # Default span for the beta search
INSERT_DISTANCE = 20  # cm entre les prises


def sort_boulder_holds(boulder):
    """ Sorts the boulder holds in a specific order."""
    return sorted(boulder, key=lambda x: ord(x[0]) - 65 + int(x[1:]) * 100)

def get_distance_pos(a: tuple, b: tuple) -> float:
    return np.linalg.norm(np.array(a) - np.array(b)) * INSERT_DISTANCE

def get_distance(a: str, b: str) -> float:
    """ Calculates the distance between two holds in cm."""
    a_row, a_col = ord(a[0]) - 65, int(a[1:]) - 1
    b_row, b_col = ord(b[0]) - 65, int(b[1:]) - 1
    return get_distance_pos((a_row, a_col), (b_row, b_col))


def load_holds_data():
    """ Loads holds data from the dataset."""
    with open(HOLDS_EVAL_PATH, 'r') as file:
        reader = csv.DictReader(file)
        dataset = {row["hold"]: row for row in reader}
    new_dataset = dataset.copy()  # Create a copy of the dataset to avoid modifying the original
    for hold in dataset:
        # Convert can_match to boolean for easier evaluation
        new_dataset[hold]["can_match"] = dataset[hold]["can_match"].lower() == "true"
        new_dataset["L" + hold] = {**dataset[hold], "difficulty": dataset[hold]["difficulty_left_hand"]}
        new_dataset["R" + hold] = {**dataset[hold], "difficulty": dataset[hold]["difficulty_right_hand"]}
        new_dataset["M" + hold] = {**dataset[hold], "difficulty": min(dataset[hold]["difficulty_right_hand"], dataset[hold]["difficulty_left_hand"])}
    return new_dataset