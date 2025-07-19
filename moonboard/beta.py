import json
import sys
import csv
import commons


""" Gandalf Beta finder
Hand rules :
1. 3 starts possible :
    - Main gauche sur la prise de gauche, main droite sur la prise de droite
    - Main droite sur la prise de gauche, main gauche sur la prise de droite
    - Les deux mains sur la prise la plus basse (si elle est suffisamment grande (can_match=True))
2. Les prises suivantes ne doivent pas déjà avoir été utilisées par la même main dans le beta
3. une prise trop loin (distance > envergure) ne peut pas être utilisée pour le prochain mouvement
4. les croisés sont possibles à une distance de envergure / 2 et seulement si l'orientation de la prise le permet (pas de croisé sur une prise inversée)

- si dans une des branche de cette arbre l'aglo ne peut rien faire la beta est mauvaise et on coupe la branche (on peut avoir une liste des betas foireuse comme ça)
- si on a une prise de fin dans la main on match et la branche est terminée

Feet rules :
1 - si tu va en main droite alors 2 possibilités tu cherche un pied droit à  droite de ta prise de main gauche actuelle en dessous des main actuelles ou un pied gauche à gauche de ta prise de main gauche actuelle, toujours le plus haut possible
2 - pareil en symetrie pour la gauche

beta de la forme liste de placements
placement de la forme dictionaire dont les clés sont MG, MD  et les valeurs sont les coordonnées des prises


TODO :
1 - Rajouter les caracteristiques des prises pour determiner :
  - les prises que l'on ne peux pas matcher
  - les envergures max et min possibles
2 - ranking des betas
"""

# https://gitlab.com/shape-team/shape-project/-/blob/master/flaskapp/gandalf.py?ref_type=heads

####################
# Functions to evaluate the type of move
######################
def is_crossing(mouv, previous_mouv):
    if not previous_mouv:
        return False
    hand = mouv[0]  # Get the hand used for the current move
    previous_hand = previous_mouv[0]
    hold = mouv[1:]  # Get the hold name without the hand prefix
    previous_hold = previous_mouv[1:]  # Get the previous hold name without the hand prefix
    return hand < previous_hand and hold[0] > previous_hold[0] or hand > previous_hand and hold[0] < previous_hold[0]

def is_match(mouv, previous_mouv):
    if not previous_mouv:
        return False
    hand = mouv[0]  # Get the hand used for the current move
    previous_hand = previous_mouv[0]
    hold = mouv[1:]  # Get the hold name without the hand prefix
    previous_hold = previous_mouv[1:]  # Get the previous hold name without the hand prefix
    return hand != previous_hand and hold == previous_hold

def is_bump(mouv, previous_mouv):
    if not previous_mouv:
        return False
    hand = mouv[0]  # Get the hand used for the current move
    previous_hand = previous_mouv[0]
    return hand == previous_hand

def filter_not_already_used_holds(boulder: list, beta: list) -> list:
    """ Filters the boulder holds to only include those that are not already used for both hands in the beta."""
    used_holds_left = [mouv[1:] for mouv in beta if mouv[0] == "L"]
    used_holds_right = [mouv[1:] for mouv in beta if mouv[0] == "R"]
    return [hold for hold in boulder if hold not in used_holds_left + used_holds_right]  # Return only the holds not used in the beta


###################
# Function to find possible beta for the next hold
###################
def find_beta_next_hold(boulder: list, beta_list: list, holds_data: dict, span: int=180) -> list[list[str]]:
    """ Finds the next possible holds for each beta in the list."""
    global iteration
    iteration+=1
    print(f"Iteration {iteration}: {len(beta_list)} betas to process")
    next_beta_list = []
    finished = True  # Flag to check if we have finished processing all betas
    for beta in beta_list:
        last_mouv = beta[-1]  # Get the last hold in the current beta
        last_hold = last_mouv[1:]  # Get the hold name without the hand prefix
        idx, previous_other_hand = find_previous_other_hand_mouv(beta)  # Find the previous move with the other hand
        if len(beta) > len(boulder) * 2 or idx < len(beta) - 3:
            # Skip betas that are too long (more than twice the number of holds in the boulder)
            # skip beta with more than 2 bumps in a row (this is a heuristic to avoid too many bumps in the beta)
            continue
        elif last_hold == boulder[-1]:
            next_beta_list.append(beta)
            continue  # If the last hold is the last hold of the boulder, add the beta as is and skip to the next beta
        # Get the next possible holds based on the last hold
        next_holds = filter_not_already_used_holds(boulder, beta)  # Filter out holds already used with both hands in the beta
        new_betas = []
        for next_hold in next_holds:
            can_match = holds_data[next_hold]['can_match']  # Check if the next hold can be matched
            orientation = holds_data[next_hold]['orientation']  # Get the orientation of the next hold
            # type_ = holds_data[next_hold]['type']  # Get the type of the next hold
            for mouv, other_mouv in [(last_mouv, previous_other_hand), (previous_other_hand, last_mouv)]:
                other_hand = other_mouv[0]
                hold = mouv[1:]
                other_hold = other_mouv[1:]
                next_mouv = other_hand + next_hold  # Create the next move with the other hand and the next hold
                if (
                    # beta + [next_mouv] in new_betas
                    # or next_mouv in beta  # If the next move is already in the beta, skip it
                    commons.get_distance(hold, next_hold) > span
                    or commons.get_distance(other_hold, next_hold) > span
                    or int(next_hold[1:]) <= min(int(other_hold[1:]), int(hold[1:]))  # If the next hold is lower than the last holds, skip it
                ):
                    continue  # If the next hold is too far, skip it
                elif (
                    (commons.get_distance(hold, next_hold) < span / 3 and is_crossing(next_mouv, mouv) and "S" not in orientation) or not is_crossing(next_mouv, mouv)  # If the next hold is close enough to the last hold and is a crossing move and the orientation is not "south" (undercling)
                ) and (  # Check if the next move is not already in the beta
                    (can_match and is_match(next_mouv, mouv)) or not is_match(next_mouv, mouv)  # If the next hold can be matched and is not already used by the same hand
                ) and ( # If the next hold can be matched and is not already used by the other hand
                    not is_bump(next_mouv, last_mouv)
                    or (is_bump(next_mouv, last_mouv) and int(next_hold[1:]) >= int(other_hold[1:])) # If the next hold is a bump move and upper than the last hold 
                ):
                    new_betas.append(beta + [next_mouv])
            # if no new betas were found, we can skip this beta as it's not valid (cannot reach the next hold)
        if new_betas:
            # If we have new betas, add them to the next_beta_list
            next_beta_list.extend(new_betas)
            finished = False  # If we found new betas, we are not finished yet

    if finished:
        return next_beta_list # If all betas are finished, return the list as is
    return find_beta_next_hold(boulder, next_beta_list, holds_data, span)


def beta_listing(boulder: list, holds_data: dict, span: int = 170) -> list[list[dict]]:
    global iteration
    iteration = 0
    """ Returns a list of possible beta for the boulder problem."""
    start_holds = boulder[:2]  # Get the first two holds for the start
    # start rule n°1
    beta_list = [
        ['L' + start_holds[0], 'R' + start_holds[1]],  # Main gauche sur la prise de gauche, main droite sur la prise de droite
        ['R' + start_holds[0], 'L' + start_holds[1]],  # Main droite sur la prise de gauche, main gauche sur la prise de droite
    ]
    sorted_start_holds = sorted(start_holds, key=lambda x: x[1:])  # Sort holds by their vertical position (1,2,3, etc.)
    if holds_data[sorted_start_holds[0]]['can_match']:
        beta_list.append(['L' + sorted_start_holds[0], 'R' + sorted_start_holds[0]]),  # les deux mains sur la première prise dans un sens
        beta_list.append(['R' + sorted_start_holds[0], 'L' + sorted_start_holds[0]]),  # les deux mains sur la première prise dans l'autre sens
    return find_beta_next_hold(boulder, beta_list, holds_data, span=span)



def filter_only_finishing_beta(beta_list: list, boulder: list):
    """ Filters the beta list to only include those that finish on the last hold of the boulder."""
    last_hold = boulder[-1]
    for beta in beta_list:
        if beta[-1][1:] == last_hold:
            yield beta


def possible_betas(boulder, holds_data, span=170) -> list[list[str]]:
    """ Returns a list of possible beta for the boulder problem."""
    beta_list = beta_listing(boulder, holds_data, span=span)
    # filtered_finished_betas = list(filter_only_finishing_beta(beta_list, boulder))
    filtered_betas = evaluate_betas_difficulty(beta_list, holds_data)  # Evaluate the difficulty of each beta
    return filtered_betas

def best_betas(betas: list[dict], max_betas: int | None = None) -> list[dict]:
    """ Returns the best beta based on the difficulty score."""
    if not betas:
        return []
    min_max_difficulty = min([beta['max'] for beta in betas])  # Get the minimum maximum difficulty in the betas
    betas = list(filter(lambda x: x['max'] == min_max_difficulty, betas)) # Filter betas with the minimum maximum difficulty
    min_number_max_difficulty =  min([beta['mouv_at_max'] for beta in betas])  # Get the minimum number of moves at maximum difficulty
    betas = list(filter(lambda x: x['mouv_at_max'] == min_number_max_difficulty, betas)) # Filter betas with the minimum maximum difficulty
    min_distance = min([beta['max_distance'] for beta in betas]) # Get the minimum maximum distance in the betas
    betas = list(filter(lambda x: x['max_distance'] == min_distance, betas))  # then filter betas with the minimum maximum distance
    min_mean = min([beta['mean'] for beta in betas])  # then Get the minimum number of crosses in the beta
    betas = list(filter(lambda x: x['mean'] == min_mean, betas)) # Filter betas with the minimum maximum difficulty
    min_mean_distance = min([beta['mean_distance'] for beta in betas])  # then Get the minimum number of crosses in the beta
    betas = list(filter(lambda x: x['mean_distance'] == min_mean_distance, betas)) # Filter betas with the minimum maximum difficulty
    # min_sum = min([beta['sum'] for beta in betas])  # then Get the minimum number of crosses in the beta
    # betas = list(filter(lambda x: x['sum'] == min_sum, betas)) # Filter betas with the minimum maximum difficulty
    # min_bumps =  min([beta['bumps'] for beta in betas])  # then Get the minimum number of bumps in the beta
    # betas = list(filter(lambda x: x['bumps'] == min_bumps, betas))  # then filter betas with the minimum number of bumps
    # min_matches = min([beta['matches'] for beta in betas])  # then Get the minimum number of matches in the betas
    # betas = list(filter(lambda x: x['matches'] == min_matches, betas))  # then filter betas with the minimum number of matches
    # min_crosses = min([beta['crosses'] for beta in betas])  # then Get the minimum number of crosses in the beta
    # betas = filter(lambda x: x['crosses'] == min_crosses, betas)
    # print(f"Found {len(betas)} betas with most difficult hand {min_max_difficulty}, matches {min_matches}, max distant mouv {min_distance}, and moves at max difficulty {min_number_max_difficulty}")
    return betas[:min(len(betas), max_betas or len(betas))]  # Return the top max_betas betas

###################
# Evaluate beta difficulty
###################

def find_previous_other_hand_mouv(beta: list) -> str | None:
    """ Finds the previous move with the other hand in the beta."""
    mouv = beta[-1]  # Get the last move in the beta
    for idx, previous_mouv in reversed(list(enumerate(beta[:-1]))):  # Exclude the last move as it is the current move
        if previous_mouv[0] != mouv[0]:  # Check if the hand is different
            return idx, previous_mouv  # Return the previous move with the other hand
    return None, None  # If no previous move with the other hand is found, return None

def evaluate_beta_difficulty(beta: list, holds_data: dict) -> dict:
    """ Evaluates the difficulty of a beta based on the holds used."""
    score = {
        "sum": 0,
        "max": 0,
        "mouv_at_max": 0,
        "mean": 0.0,
        "mouvs": len(beta) - 2,  # Number of moves in the beta (excluding start position)
        "matches": 0,
        "crosses": 0,
        "bumps": 0,
        "max_distance": 0,
        "mouvs_at_max_distance": 0,
        "sum_distance": 0,
    }
    for index, mouv in enumerate(beta):
        hold_data = holds_data[mouv]  # Get the hold data without the hand prefix
        difficulty = int(hold_data['difficulty'])
        previous_mouv = beta[index - 1] if index > 0 else None
        idx, previous_other_hand = find_previous_other_hand_mouv(beta[:index+1])  # Find the previous move with the other hand
        distance = round(commons.get_distance(previous_other_hand[1:], mouv[1:])) if previous_other_hand else 0  # Get the distance between the previous hold and the current hold
        if is_match(mouv, previous_mouv):
            score["matches"] += 1
        if is_crossing(mouv, previous_mouv):
            score["crosses"] += 1
        if is_bump(mouv, previous_mouv):
            score["bumps"] += 1
        score["sum"] += difficulty
        score['sum_distance'] += distance  # Add the distance to the sum
        if difficulty > score["max"]:
            score["max"] = difficulty
            score["mouv_at_max"] = 1
        elif difficulty == score["max"]:
            score["mouv_at_max"] += 1
        if distance > score["max_distance"]:
            score["max_distance"] = distance
            score["mouvs_at_max_distance"] = 1
        elif distance == score["max_distance"]:
            score["mouvs_at_max_distance"] += 1
    score["mean"] = round(float(score["sum"]) / score["mouvs"]) if score["mouvs"] > 0 else 0
    score["mean_distance"] = round(float(score["sum_distance"]) / score["mouvs"]) if score["mouvs"] > 0 else 0
    return score

def evaluate_betas_difficulty(betas: list, holds_data: dict) -> list[dict]:
    """ Evaluates the difficulty of each beta in the list."""
    return [{**evaluate_beta_difficulty(beta, holds_data), "beta": beta} for beta in betas] # Evaluate each beta in the list and return the results

#################
# Main function to find possible betas for a given boulder
###################
def main(boulder: str):
    boulder = json.loads(boulder.replace("'", "\"")) # expected format: ["A1", "B2", "C3"]
    boulder = commons.sort_boulder_holds(boulder)
    holds_data = commons.load_holds_data()
    betas = possible_betas(boulder, holds_data, span=180)
    best_betas_list = best_betas(betas, max_betas=3)  # Get the best betas based on the difficulty score
    print(f"Iterations {iteration}: {len(betas)} betas found, {len(best_betas_list)} best betas found")
    return best_betas_list

    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python beta.py <boulder>")
        sys.exit(1)
    print(main(*sys.argv[1:]))
