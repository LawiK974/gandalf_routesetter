#!/usr/bin/python
#-*- coding: utf-8 -*-

import random
import numpy as np
import scipy.stats as stats
try:
    import commons
except:
    from . import commons

SIZE = (18, 11)  # moonboard classique
# SIZE = (12, 11)  # mini moonboard
INSERT_DISTANCE = 20  # cm entre les prises
board = [[(x, y) for y in range(SIZE[1])] for x in range(SIZE[0])]
ENVERGURE = 170


def get_weight_gaussian(distance: float, mu: float = (ENVERGURE - 22)/2, sigma: float = (ENVERGURE - 22)/4) -> float:
    """ Returns a weight according to a gaussian distribution."""
    # par défaut mu est la moitié de l'envergure
    # et sigma est un quart de l'envergure (68% des prises seront entre mu-sigma et mu+sigma)
    return float(stats.norm.pdf(distance, loc=mu, scale=sigma))


def get_boulder(span: int = ENVERGURE, hold_types: list = None) -> list[str]:
    """ Converts a list of holds to a string representation. Optionally filter by hold_types."""
    holds_data = commons.load_holds_data() if hold_types else None
    boulder = get_next_hold(span=span, hold_types=hold_types, holds_data=holds_data)
    return [chr(65 + y) + str(x+1) for x,y in boulder]


def get_next_hold(previous: tuple[int, int] | None = None, span: int=ENVERGURE, hold_types: list = None, holds_data: list = None, start: bool = False) -> list[tuple[int, int]]:
    """ Returns the next hold to grab based on the previous hold, optionally filtering by hold_types."""
    if previous is None:
        # prise de départ au hasard en dessous de la 6eme rangee
        candidates = board[:6]
        if hold_types and holds_data:
            candidates = [[hold for hold in row if holds_data.get(commons.get_hold_name(hold), {})["type"] in hold_types] for row in candidates]
            if not candidates:
                raise ValueError("No holds found matching the specified types.")
        hold = random.choice(random.choice(candidates))
        start = not holds_data[commons.get_hold_name(hold)]["can_match"]
    elif previous[0] == SIZE[0] - 1:
        # fin du mur
        return []
    else:
        possible_holds = []
        weight_list = []
        mu = (span - 22) / 2
        sigma = mu / 2
        for row in board[:6] if start else board[previous[0] + 1:]:
            for new in row:
                distance = commons.get_distance_pos(previous, new)
                # restriction a uniquement les prises au dessus de la precedente
                # et a une distance inferieure à l'envergure
                if distance < (span - 22):
                    hold_name = commons.get_hold_name(new)
                    if hold_types and holds_data:
                        undercling = 'S' in holds_data[hold_name]["orientation"]
                        if holds_data[hold_name]["type"] not in hold_types or (undercling and 'undercling' not in hold_types):
                            continue
                    possible_holds.append(new)
                    weight_list.append(get_weight_gaussian(distance, mu, sigma))  # poids selon la distance
        if not possible_holds:
            return []
        hold = random.choices(possible_holds, weight_list, k=1)[0]
        start = False
    # on ajoute la prise courante à la liste des prises
    return [hold] + get_next_hold(hold, span, hold_types, holds_data, start=start)


def main():
    """ Main function to generate a boulder problem."""
    boulder = get_boulder()
    boulder = commons.sort_boulder_holds(boulder)
    print(f"Generated Boulder: {boulder}")

if __name__ == "__main__":
    main()
