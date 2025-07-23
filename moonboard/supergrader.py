#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import json
try:
    import commons
    import grader
except:
    from . import commons, grader

def loading_dataset():
    print("Loading dataset...")
    dataset = []
    holds_data = {}
    for version in ["2019", "2016", "2017"]:
        print(f"Loading holds data for version {version}...")
        holds_data[version] = commons.load_holds_data(version=version)
        dataset.extend(commons.load_boulders_from_dataset(version=version))
    filtered_dataset = grader.filter_dataset(dataset)
    print(f"Filtered dataset now {len(filtered_dataset)} boulders")
    dataset = grader.BoulderDataset(filtered_dataset, holds_data)
    return dataset

def main(phase, boulder_json = None, model_path = None, boulder_object = None):
    input_size = 12  # Default input size for the model
    if phase == "train":
        dataset = loading_dataset()
        grader.train_model(dataset, hidden_size=512, batch_size=64, learning_rate=1e-3, weight_decay_factor=0, num_epochs=52, focal_gamma=5.0, focal_alpha=0.75, early_stopping=True, input_size=input_size)
    elif phase == "predict":
        boulder = boulder_object
        if boulder_json:
            boulder = json.load(boulder_json)
        pred_grade, prob_percent = grader.predict_boulder_grade(boulder, model_path, input_size=input_size, version=boulder["version"])
        if boulder_object:
            return pred_grade, prob_percent
        print(f"Predicted grade: {pred_grade}, Probability: {prob_percent:.2f}%")
    elif phase == "search":
        dataset = loading_dataset()
        results, best_config = grader.hyperparameter_search(dataset, input_size=input_size)

if __name__ == "__main__":
    main(*sys.argv[1:])