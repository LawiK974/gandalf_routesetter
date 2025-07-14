#!/usr/bin/python
#-*- coding: utf-8 -*-

import random
import numpy as np
import scipy.stats as stats
from . import commons
from . import similar_boulder as sb

SIZE = (18, 11)  # moonboard classique
# SIZE = (12, 11)  # mini moonboard
INSERT_DISTANCE = 20  # cm entre les prises
board = [[(x,y) for y in range(SIZE[1])] for x in range(SIZE[0])]
ENVERGURE = 170

def get_distance(a: tuple,b: tuple) -> float:
    """ Calculates the distance between two holds in cm."""
    return np.linalg.norm(np.array(a) - np.array(b)) * INSERT_DISTANCE

def get_weight_gaussian(distance: float, mu: float = ENVERGURE/2, sigma: float = ENVERGURE/4) -> float:
    """ Returns a weight according to a gaussian distribution."""
    # par défaut mu est la moitié de l'envergure
    # et sigma est un quart de l'envergure (68% des prises seront entre mu-sigma et mu+sigma)
    return stats.norm.pdf(distance, loc=mu, scale=sigma)

def get_boulder(span: int = ENVERGURE) -> list[str]:
    """ Converts a list of holds to a string representation."""
    boulder = get_next_hold(span=span)
    return [chr(65 + y) + str(x+1) for x,y in boulder]

def get_next_hold(previous: tuple[int, int] | None = None, span: int=ENVERGURE) -> list[tuple[int, int]]:
    """ Returns the next hold to grab based on the previous hold."""
    if previous is None:
        # prise de départ au hasard en dessous de la 6eme rangee
        hold = random.choice(random.choice(board[:6]))
    elif previous[0] == SIZE[0] - 1:
        # fin du mur
        return []
    else:
        possible_holds = []
        weight_list = []
        for row in board[previous[0]+1:]:
            for new in row:
                distance =  get_distance(previous, new)
                # restriction a uniquement les prises au dessus de la precedente 
                # et a une distance inferieure à l'envergure
                if distance <= ENVERGURE:
                    possible_holds.append(new)
                    weight_list.append(get_weight_gaussian(distance, span/2, span/4))  # poids selon la distance
        hold = random.choices(possible_holds, weight_list, k=1)[0]
    
    # on ajoute la prise courante à la liste des prises
    return [hold] + get_next_hold(hold, span)

def main():
    """ Main function to generate a boulder problem."""
    boulder = get_boulder()
    print(f"Generated Boulder: {boulder}")
    similar_boulder, score = sb.similar_boulders(boulder, sb.load_boulders_from_dataset(commons.DATASET_PATH))
    print(f"Most similar Boulder: {similar_boulder}, Score: {score*100:.2f}%")

if __name__ == "__main__":
    main()
