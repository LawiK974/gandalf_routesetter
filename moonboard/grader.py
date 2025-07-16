#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np


def vectorize_beta(beta, holds_data=None):
    """Convertit une beta en vecteur de difficulté des prises utilisées."""
    if holds_data is None:
        raise ValueError("holds_data doit être fourni")
    vector = np.zeros(commons.SIZE)  # On suppose que la taille du mur est de 11 colonnes (A à K)
    for mouv in beta:
        hold_name = mouv[1:]
        hold_pos = commons.hold_name_to_pos(hold_name)
        vector[idx] = float(holds_data[mouv]["difficulty"])
    return vector

def filter_dataset(dataset):
    """Filtre le dataset pour ne garder que les boulders avec des prises valides."""
    list_benchmark_setters = set()
    for boulder in dataset:
        if boulder['isBenchmark'] and boulder['setter'] not in list_benchmark_setters:
            list_benchmark_setters.add(boulder['setter'])
    filtered_dataset = list(filter(lambda boulder: (
        boulder['method'] == "Feet follow hands"
        and boulder["userGrade"] is not None
        and (
            boulder['repeats'] > 20
            or boulder['setter'] in list_benchmark_setters
        )
    ), dataset))
    print(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} boulders.")
    return filtered_dataset

def main():
    holds_data = commons.load_holds_data()
    dataset = commons.load_boulders_from_dataset()
    filtered_dataset = filter_dataset(dataset)
    
    

if __name__ == "__main__":
    import commons
    main()
else:
    from . import commons