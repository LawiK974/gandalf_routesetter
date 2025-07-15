import os
import numpy as np
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