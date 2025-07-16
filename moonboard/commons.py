import os
import numpy as np
import csv
import json
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/problems.json")
HOLDS_EVAL_PATH = os.path.join(SCRIPT_PATH, "../dataset/2019.csv")
ENVERGURE = 170  # Default span for the beta search
INSERT_DISTANCE = 20  # cm entre les prises
SIZE = (18, 11)  # moonboard classique

GRADE_MAP = {
    "6A+": 0,
    "6B": 1,
    "6B+": 2,
    "6C": 3,
    "6C+": 4,
    "7A": 5,
    "7A+": 6,
    "7B": 7,
    "7B+": 8,
    "7C": 9,
    "7C+": 10,
    "8A": 11,
    "8A+": 12,
    "8B": 13,
    "8B+": 14
}


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
    return new_dataset

def get_hold_name(hold: tuple[int, int]) -> str:
    """ Returns the name of a hold given its coordinates."""
    return chr(65 + hold[1]) + str(hold[0] + 1)

def hold_name_to_pos(hold_name: str) -> tuple[int, int]:
    """ Converts a hold name to its position on the board."""
    col = ord(hold_name[0]) - 65
    row = int(hold_name[1:]) - 1
    return (row, col)

def load_boulders_from_dataset():
    """Load boulders from a JSON file."""
    boulder_list = []
    with open(DATASET_PATH, 'r') as file:
        dataset = json.load(file)
    for boulder in dataset["data"]:
        boulder_list.append({
            "holds": sort_boulder_holds([hold["description"] for hold in boulder["moves"]]),
            "name": boulder["name"],
            "setter": boulder["setby"],
            "grade": boulder["grade"],
            "userGrade": boulder["userGrade"],
            "rating": boulder["userRating"],
            "repeats": boulder["repeats"],
            "isBenchmark": boulder["isBenchmark"],
            "method": boulder["method"],
        })
    return boulder_list