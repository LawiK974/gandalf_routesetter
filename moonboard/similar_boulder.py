import json
import sys
import commons

def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def similar_boulders(boulder, boulder_list):
    max_score = -1
    similar_boulder = None
    for other_boulder in boulder_list:
        score = jaccard_similarity(set(sorted(boulder)), set(sorted(other_boulder["holds"])))
        if score > max_score:
            max_score = score
            similar_boulder = other_boulder
    return similar_boulder, max_score


def load_boulders_from_dataset(file_path):
    """Load boulders from a JSON file."""
    boulder_list = []
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    for boulder in dataset["data"]:
        boulder_list.append({
            "holds": [hold["description"] for hold in boulder["moves"]],
            "name": boulder["name"],
            "setter": boulder["setby"],
            "grade": boulder["grade"],
            "userGrade": boulder["userGrade"],
        })
    return boulder_list

def main(boulder: str):
    boulder = json.loads(boulder.replace("'", "\"")) # expected format: ["A1", "B2", "C3"]
    boulder_list = load_boulders_from_dataset(commons.DATASET_PATH)
    similar_boulder, score = similar_boulders(boulder, boulder_list)

    if similar_boulder:
        return f"Most Similar Boulder: {similar_boulder}, Score: {score}"
    else:
        return "No similar boulder found."

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python similar_boulder.py <boulder>")
        sys.exit(1)
    print(main(*sys.argv[1:]))