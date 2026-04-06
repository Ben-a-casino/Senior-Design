import random
import pandas as pd


# =============================================================================
# FUNCTION: apply_changes
# PURPOSE:
#   Combines the original routes with the change matrix (a matrix of mostly
#   -1s) to produce the current (mutated) version of the routes.
#
# INPUTS:
#   original_routes : list[list[str]]  — the unmodified starting routes
#   change_matrix   : list[list]       — same shape as original_routes;
#                                        -1 means no change, any other value
#                                        is the mutated tract ID
# OUTPUT:
#   effective_routes : list[list[str]] — routes with all mutations applied
#
# LOGIC:
#   - If change_matrix[r][s] == -1 → keep original node
#   - Otherwise → replace with mutated node
# =============================================================================
def apply_changes(original_routes, change_matrix):

    effective_routes = []  # will store updated routes

    # loop through each route
    for r in range(len(original_routes)):
        effective_route = []

        # loop through each stop in the route
        for s in range(len(original_routes[r])):

            # if no mutation at this position, keep original
            if change_matrix[r][s] == -1:
                effective_route.append(original_routes[r][s])
            else:
                # otherwise use mutated value
                effective_route.append(change_matrix[r][s])

        # add completed route
        effective_routes.append(effective_route)

    return effective_routes


# =============================================================================
# FUNCTION: valid_replacements
# PURPOSE:
#   Finds all valid tract IDs that can replace a specific stop in a route.
#
# INPUTS:
#   original_routes : list[list[str]]  — the unmodified starting routes
#   change_matrix   : list[list]       — current mutation state (mostly -1s)
#   r               : int              — index of the route being checked
#   s               : int              — index of the stop being checked
#   adj             : pd.DataFrame     — binary adjacency matrix; adj.loc[i,j]
#                                        == 1 means tract i and j are neighbors
#                                        tract IDs are both the index and columns
#   depot           : str              — depot tract ID; excluded from all routes
#
# OUTPUT:
#   list[str] — tract IDs that are valid replacements for stop s on route r
#
# CONSTRAINTS:
#   - Cannot use the depot (the start/end of a route)
#   - Cannot reuse nodes already used in any route
#   - Must be adjacent to the previous stop in the route (if one exists)
#   - Must be adjacent to the next stop in the route (if one exists)
# =============================================================================
def valid_replacements(original_routes, change_matrix, r, s, adj, depot):

    # reconstruct current routes after mutations
    routes = apply_changes(original_routes, change_matrix)
    route = routes[r]

    old = route[s]  # current node at this position

    # collect all nodes currently used across all routes
    used = {node for rt in routes for node in rt}
    used.remove(old)  # allow replacement at this position

    # initial candidate pool:
    # all nodes minus used nodes and depot
    candidates = set(adj.index) - used - {depot}

    """
    Diran here, I believe the initial candidates should
    be any node the current node is adjacent to, based
    on the adjacency matrix. You can get that set using:
    
        candidates = set(np.flatnonzero(adj[old,:]))

    np.flatnonzero returns the indices of nonzero elements 
    of the input array. Since the adjacency matrix is zeros
    and ones, life is good. 

    The row of the adjacency matrix corresponding to the 
    current node indicates all other nodes adjacent to it,
    hence slicing that row and all columns. 
    """
    
    # enforce adjacency with previous node (if exists)
    if s > 0:
        prev_node = route[s - 1]
        candidates = {c for c in candidates if adj.loc[prev_node, c] == 1}

    # enforce adjacency with next node (if exists)
    if s < len(route) - 1:
        next_node = route[s + 1]
        candidates = {c for c in candidates if adj.loc[next_node, c] == 1}

    return list(candidates)


# =============================================================================
# FUNCTION: mutate
# PURPOSE:
#   Applies up to delta_max mutations to the routes while respecting the
#   mutation budget, adjacency constraints, and node uniqueness.
#
# INPUTS:
#   original_routes : list[list[str]]  — the unmodified starting routes;
#                                        never modified by this function
#   change_matrix   : list[list]       — current mutation state passed in from
#                                        the GA loop; must be initialized to
#                                        all -1s before the very first call:
#                                        [[-1]*len(route) for route in original_routes]
#                                        after the first call, pass the returned
#                                        cm back in to accumulate changes across
#                                        generations
#   adj             : pd.DataFrame     — binary adjacency matrix; tract IDs are
#                                        both the index and columns
#   depot           : str              — depot tract ID; excluded from all routes
#   delta_max       : int              — maximum number of stops that may differ
#                                        from original_routes at any point;
#                                        already-mutated stops can be re-mutated
#                                        without consuming extra budget
#
# OUTPUT:
#   mutated_routes  : list[list[str]]  — final routes after mutation
#   cm              : list[list]       — updated change matrix; pass this back
#                                        into mutate on the next generation
#   changes         : list[tuple]      — one entry per mutation made:
#                                        (route_index, stop_index, old_tract, new_tract)
# =============================================================================
def mutate(original_routes, change_matrix, adj, depot, delta_max):

    # copy change matrix so the caller's copy is not modified
    cm = [row[:] for row in change_matrix]

    changes = []  # track mutations made this call

    # attempt up to delta_max mutations
    for _ in range(delta_max):

        # count how many positions already differ from original
        current_changes = sum(1 for row in cm for val in row if val != -1)

        feasible = []  # store valid (r, s) positions to mutate

        # scan all positions in all routes
        for r in range(len(original_routes)):
            for s in range(len(original_routes[r])):

                # a pristine position (-1) costs one budget slot;
                # skip it if the budget is already full
                if cm[r][s] == -1 and current_changes >= delta_max:
                    continue

                # only include positions with at least one valid replacement
                if valid_replacements(original_routes, cm, r, s, adj, depot):
                    feasible.append((r, s))

        # stop early if no valid mutation positions exist
        if not feasible:
            break

        # randomly choose a position to mutate
        r, s = random.choice(feasible)

        # get valid replacement options for that position
        choices = valid_replacements(original_routes, cm, r, s, adj, depot)

        # randomly choose a replacement tract
        new_tract = random.choice(choices)

        # record the tract being replaced before applying the mutation
        old_tract = apply_changes(original_routes, cm)[r][s]

        # apply mutation to the change matrix
        cm[r][s] = new_tract

        # store mutation details
        changes.append((r, s, old_tract, new_tract))

    # reconstruct final routes from original + updated change matrix
    mutated_routes = apply_changes(original_routes, cm)

    return mutated_routes, cm, changes


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
# delta_max = 2 (at most 2 stops may differ from original across all generations)
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
    print(f"delta_max: {delta_max}  (max positions that may differ from original)")
    print("=" * 50)

    print("\n--- Generation 0 (original) ---")
    print("Routes:        ", original_routes)
    print("Change matrix: ", change_matrix)

    # ── Generation 1 ──────────────────────────────────────────────────────────
    mutated_routes, change_matrix, changes = mutate(
        original_routes, change_matrix, adj, depot, delta_max
    )
    print("\n--- Generation 1 ---")
    print("Routes:        ", mutated_routes)
    print("Change matrix: ", change_matrix)
    print("Changes:       ", changes)
    total = sum(1 for row in change_matrix for v in row if v != -1)
    print(f"Total diffs from original: {total}/{delta_max}")

    # ── Generation 2 ──────────────────────────────────────────────────────────
    # pass the returned change_matrix back in to carry the budget forward
    mutated_routes, change_matrix, changes = mutate(
        original_routes, change_matrix, adj, depot, delta_max
    )
    print("\n--- Generation 2 ---")
    print("Routes:        ", mutated_routes)
    print("Change matrix: ", change_matrix)
    print("Changes:       ", changes)
    total = sum(1 for row in change_matrix for v in row if v != -1)
    print(f"Total diffs from original: {total}/{delta_max}")

    # ── Generation 3 ──────────────────────────────────────────────────────────
    mutated_routes, change_matrix, changes = mutate(
        original_routes, change_matrix, adj, depot, delta_max
    )
    print("\n--- Generation 3 ---")
    print("Routes:        ", mutated_routes)
    print("Change matrix: ", change_matrix)
    print("Changes:       ", changes)
    total = sum(1 for row in change_matrix for v in row if v != -1)
    print(f"Total diffs from original: {total}/{delta_max}")
'''
