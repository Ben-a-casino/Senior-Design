import random              
import numpy as np        
import pandas as pd       


# ============================================================
# APPLY CHANGES
# ------------------------------------------------------------
# Combines the original routes with the change_matrix to produce
# the current "mutated" version of routes.
#
# change_matrix stores mutations:
#   -1  → keep original node
#   else → replace with that node
# ============================================================
def apply_changes(original_routes, change_matrix):

    effective_routes = []  # final list of updated routes

    # loop over each route
    for r in range(len(original_routes)):
        effective_route = []  # build one route at a time

        # loop over each stop in that route
        for s in range(len(original_routes[r])):

            # if no mutation → keep original node
            if change_matrix[r][s] == -1:
                effective_route.append(original_routes[r][s])

            # otherwise → use mutated node
            else:
                effective_route.append(change_matrix[r][s])

        # store completed route
        effective_routes.append(effective_route)

    return effective_routes


# ============================================================
# VALID REPLACEMENTS
# ------------------------------------------------------------
# Finds all valid nodes that can replace a specific stop.
#
# A candidate must:
#   - not be the depot
#   - not already be used elsewhere
#   - be adjacent to:
#       • the current node
#       • previous node (if exists)
#       • next node (if exists)
# ============================================================
def valid_replacements(original_routes, change_matrix, r, s, adj, depot):

    # rebuild current routes including all mutations
    routes = apply_changes(original_routes, change_matrix)
    route = routes[r]

    old = route[s]  # current node we want to replace

    # get all nodes currently used across all routes
    used = {node for rt in routes for node in rt}

    # allow reuse of this position (remove current node)
    used.remove(old)

    # convert adjacency matrix to numpy for faster indexing
    adj_array = adj.to_numpy()
    tract_ids = list(adj.index)

    # find index of current node
    old_idx = tract_ids.index(old)

    # get all neighbors of current node
    neighbors = {tract_ids[i] for i in np.flatnonzero(adj_array[old_idx, :])}

    # initial candidates:
    # neighbors of current node, excluding used nodes and depot
    candidates = neighbors - used - {depot}

    # enforce adjacency with previous node
    if s > 0:
        prev_node = route[s - 1]
        prev_idx = tract_ids.index(prev_node)

        # neighbors of previous node
        prev_neighbors = {
            tract_ids[i] for i in np.flatnonzero(adj_array[prev_idx, :])
        }

        # keep only nodes that are also neighbors of previous
        candidates = candidates.intersection(prev_neighbors)

    # enforce adjacency with next node
    if s < len(route) - 1:
        next_node = route[s + 1]
        next_idx = tract_ids.index(next_node)

        # neighbors of next node
        next_neighbors = {
            tract_ids[i] for i in np.flatnonzero(adj_array[next_idx, :])
        }

        # keep only nodes that are also neighbors of next
        candidates = candidates.intersection(next_neighbors)

    return list(candidates)  # return as list for random.choice


# ============================================================
# MUTATE
# ------------------------------------------------------------
# Performs EXACTLY ONE mutation per call.
#
# Strategy:
#   - randomly pick a route
#   - respect delta_max (max allowed changes per route)
#   - either:
#       • add/change a mutation
#       • or revert a mutation
#
# Stops when one valid mutation is made or none are possible.
# ============================================================
def mutate(original_routes, change_matrix, adj, depot, delta_max):

    # deep copy so original matrix isn't modified externally
    cm = [row[:] for row in change_matrix]

    num_routes = len(original_routes)

    # counts how many routes fail to produce a mutation
    failed_attempts = 0

    # try until all routes fail
    while failed_attempts < num_routes:

        # randomly choose a route
        r = random.randrange(num_routes)

        # count how many mutations already exist in this route
        route_changes = sum(1 for val in cm[r] if val != -1)

        route_len = len(original_routes[r])

        # ====================================================
        # CASE 1: route is already at delta_max
        # only allowed action → revert a mutation
        # ====================================================
        if route_changes >= delta_max:

            revertible = []  # stops that can safely revert

            for s in range(route_len):

                if cm[r][s] != -1:  # only mutated stops can revert

                    # rebuild routes with current mutations
                    current_routes = apply_changes(original_routes, cm)

                    original_tract = original_routes[r][s]

                    # check if original node is used elsewhere
                    used_elsewhere = {
                        node
                        for i, rt in enumerate(current_routes)
                        for j, node in enumerate(rt)
                        if not (i == r and j == s)
                    }

                    # valid revert only if original node unused elsewhere
                    if original_tract not in used_elsewhere:
                        revertible.append(s)

            # if no valid revert → skip route
            if not revertible:
                continue

            # choose a stop to revert
            s = random.choice(revertible)

            old_tract = cm[r][s]

            cm[r][s] = -1  # revert to original

            mutated_routes = apply_changes(original_routes, cm)

            return mutated_routes, cm, (r, s, old_tract, -1)

        # ====================================================
        # CASE 2: route is under delta_max
        # can mutate freely
        # ====================================================

        # shuffle stops so mutation attempts are random
        stop_indices = list(range(route_len))
        random.shuffle(stop_indices)

        for s in stop_indices:

            # --------------------------------------------
            # CASE 2A: already mutated → can change or revert
            # --------------------------------------------
            if cm[r][s] != -1:

                # get valid forward replacements
                forward_options = valid_replacements(
                    original_routes, cm, r, s, adj, depot
                )

                # check if revert is valid
                current_routes = apply_changes(original_routes, cm)
                original_tract = original_routes[r][s]

                used_elsewhere = {
                    node
                    for i, rt in enumerate(current_routes)
                    for j, node in enumerate(rt)
                    if not (i == r and j == s)
                }

                can_revert = original_tract not in used_elsewhere

                # if no valid move → skip
                if not forward_options and not can_revert:
                    continue

                # combine forward moves + optional revert
                options = forward_options + ([-1] if can_revert else [])

                chosen = random.choice(options)

                old_tract = cm[r][s]

                cm[r][s] = chosen  # update mutation

                mutated_routes = apply_changes(original_routes, cm)

                return mutated_routes, cm, (r, s, old_tract, chosen)

            # --------------------------------------------
            # CASE 2B: pristine stop → only forward mutation
            # --------------------------------------------
            else:

                forward_options = valid_replacements(
                    original_routes, cm, r, s, adj, depot
                )

                if not forward_options:
                    continue

                new_tract = random.choice(forward_options)

                old_tract = original_routes[r][s]

                cm[r][s] = new_tract

                mutated_routes = apply_changes(original_routes, cm)

                return mutated_routes, cm, (r, s, old_tract, new_tract)

        # no mutation possible on this route
        failed_attempts += 1

    # if ALL routes fail → return unchanged
    mutated_routes = apply_changes(original_routes, cm)
    return mutated_routes, cm, None

'''
# =============================================================================
# DEMO / TESTING
# PURPOSE:
#   Runs a multi-generation mutation example.
#
# NETWORK (9 tracts + 1 depot):
#
#   DEPOT --- T1 --- T2 --- T3
#              |      |      |
#             T4 --- T5 --- T6
#              |      |      |
#             T7 --- T8 --- T9
#
# Route 0: T1 → T2 → T3  (top row)
# Route 1: T7 → T8 → T9  (bottom row)
# Depot: DEPOT (cannot appear in any route)
# delta_max = 2 (at most 2 stops per route may differ from original)
# =============================================================================
if __name__ == "__main__":

    # ── Adjacency matrix ──────────────────────────────────────────────────────
    nodes = ["DEPOT", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

    edges = [
        ("DEPOT", "T1"),
        ("T1", "T2"), ("T2", "T3"),
        ("T1", "T4"), ("T2", "T5"), ("T3", "T6"),
        ("T4", "T5"), ("T5", "T6"),
        ("T4", "T7"), ("T5", "T8"), ("T6", "T9"),
        ("T7", "T8"), ("T8", "T9"),
    ]

    adj = pd.DataFrame(0, index=nodes, columns=nodes)
    for a, b in edges:
        adj.loc[a, b] = 1
        adj.loc[b, a] = 1

    depot = "DEPOT"

    original_routes = [
        ["T1", "T2", "T3"],
        ["T7", "T8", "T9"],
    ]

    # initialize change matrix before the first call
    change_matrix = [[-1] * len(route) for route in original_routes]

    delta_max = 2

    print("=" * 50)
    print("NETWORK")
    print("  DEPOT --- T1 --- T2 --- T3")
    print("             |      |      |")
    print("            T4 --- T5 --- T6")
    print("             |      |      |")
    print("            T7 --- T8 --- T9")
    print()
    print(f"Depot:     {depot}")
    print(f"delta_max: {delta_max}  (max stops per route that may differ from original)")
    print("=" * 50)

    print("\n--- Generation 0 (original) ---")
    print("Routes:        ", original_routes)
    print("Change matrix: ", change_matrix)

    # run 6 mutation steps (outer GA loop controls this)
    for gen in range(1, 7):
        mutated_routes, change_matrix, change = mutate(
            original_routes, change_matrix, adj, depot, delta_max
        )
        print(f"\n--- Generation {gen} ---")
        print("Routes:        ", mutated_routes)
        print("Change matrix: ", change_matrix)
        print("Change:        ", change)
        for r in range(len(change_matrix)):
            total = sum(1 for v in change_matrix[r] if v != -1)
            print(f"  Route {r} diffs from original: {total}/{delta_max}")
        '''
