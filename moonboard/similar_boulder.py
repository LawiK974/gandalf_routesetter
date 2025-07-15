import json
import sys

def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def similar_boulders(boulder, boulder_list):
    max_score = -1
    similar_boulders = []
    for other_boulder in boulder_list:
        score = jaccard_similarity(set(boulder), set(other_boulder["holds"]))
        other_boulder["score"] = score
        similar_boulders.append(other_boulder)
        if score > max_score:
            max_score = score
    similar_boulders = [{k: v for k,v in similar_boulder.items() if k != "score"} for similar_boulder in similar_boulders if similar_boulder["score"] == max_score]
    # similar_boulders = filter(lambda x: x["score"] == max_score, similar_boulders)
    return similar_boulders, max_score


def load_boulders_from_dataset(file_path):
    """Load boulders from a JSON file."""
    boulder_list = []
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    for boulder in dataset["data"]:
        boulder_list.append({
            "holds": commons.sort_boulder_holds([hold["description"] for hold in boulder["moves"]]),
            "name": boulder["name"],
            "setter": boulder["setby"],
            "grade": boulder["grade"],
            "userGrade": boulder["userGrade"],
            "rating": boulder["userRating"],
            "repeats": boulder["repeats"],
        })
    return boulder_list

def main(boulder: str):
    boulder = json.loads(boulder.replace("'", "\"")) # expected format: ["A1", "B2", "C3"]
    boulder = commons.sort_boulder_holds(boulder)
    boulder_list = load_boulders_from_dataset(commons.DATASET_PATH)
    similar_boulder, score = similar_boulders(boulder, boulder_list)

    if similar_boulder:
        return f"Most Similar Boulder: {similar_boulder}, Score: {score}"
    else:
        return "No similar boulder found."

if __name__ == "__main__":
    import commons
    if len(sys.argv) != 2:
        print("Usage: python similar_boulder.py <boulder>")
        sys.exit(1)
    print(main(*sys.argv[1:]))
else:
    from . import commons