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

def get_hold_difficulty(hold: str, hand: str, hold_data: dict) -> int:
    """ Returns the difficulty of a hold based on its name and hand."""
    orientation_point = {"N": 0, "NE": 1 if hand == "L" else 0, "E": 10 if hand == "L" else 5, "SE": 5 if hand == "L" else 3, "S": 4, "SW": 5 if hand == "R" else 3, "W": 3, "NW": 1 if hand == "R" else 0}
    texture_points = {"woodc": 1, "white": 2, "black": 3, "woodb": 4, "wooda": 5, "yellow": 6}
    type_points = {"jug": 0, "pinch": 1, "microjug": 2, "micropinch": 3, "crimp": 4, "sloper": 5, "pocket": 6}
    can_match_points = 0 if hold_data["can_match"].lower() == "true" else 1
    max_points = 1 + max(orientation_point.values()) + max(texture_points.values()) + max(type_points.values()) 
    # max_points = 1 + max(type_points.values()) * (max(orientation_point.values()) + max(texture_points.values()))  
    hold_difficulty = can_match_points + type_points[hold_data["type"]] + orientation_point[hold_data["orientation"]] + texture_points[hold_data["texture"]]
    # hold_difficulty = can_match_points + type_points[hold_data["type"]] * (orientation_point[hold_data["orientation"]] + texture_points[hold_data["texture"]])
    return hold_difficulty / max_points * 14 + 1  # Normalize to a scale of 1 to 15, as per the grading system

def load_holds_data():
    """ Loads holds data from the dataset."""
    with open(HOLDS_EVAL_PATH, 'r') as file:
        reader = csv.DictReader(file)
        dataset = {row["hold"]: row for row in reader}
    new_dataset = dataset.copy()  # Create a copy of the dataset to avoid modifying the original
    for hold in dataset:
        # Convert can_match to boolean for easier evaluation
        new_dataset[hold]["can_match"] = dataset[hold]["can_match"].lower() == "true"
        new_dataset["L" + hold] = {**dataset[hold], "difficulty": get_hold_difficulty(hold, "L", dataset[hold])}
        new_dataset["R" + hold] = {**dataset[hold], "difficulty": get_hold_difficulty(hold, "R", dataset[hold])}
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
        starting_holds = [hold["description"] for hold in boulder["moves"] if hold["isStart"]]
        boulder_list.append({
            # removing feets for grader
            "holds": sort_boulder_holds([hold["description"] for hold in boulder["moves"] if all(int(hold["description"][1:]) >= int(start[1:]) for start in starting_holds)]),
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