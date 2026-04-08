import numpy as np

def uniform_crossover(parents, num_children):
    # parents = list[list[list[int]]]
        # list of change matrices selected from the generation

    """
    This crossover function takes one route from each parent
    at random and assigns it to the child, same idea as in 
    PyGAD's uniform crossover function
    """

    generation = []

    # If the same parent is selected to be the father and mother I don't care, shouldn't be common and therefore matter
    random_parent_selection = np.random.randint(low = 0, high = len(parents), size = (num_children,2))
        
    # Choose which parent to take each gene from for each child
    random_crossover_selection = np.random.randint(low = 0, high = 2, size = (num_children, len(parents[0])))

    for n in range(num_children):
        father, mother = (parents[ind] for ind in random_parent_selection[n])
        child = []
        for i, binary in enumerate(random_crossover_selection[n]):
            if binary == 0:
                child.append(father[i])
            else:
                child.append(mother[i])
        generation.append(child)

    return generation

def spread_crossover(parents, num_children, ga_instance):
    # parents = list[list[list[int]]]
        # list of change matrices selected from the generation
    # num_children = int
        # self_explanatory
    
    """
    Similar to uniform crossover, but instead of selecting
    two parents to choose genes from, use all parents.
    """

    generation = []

    random_gene_selection = np.random.randint(low = 0, high = len(parents), size = (num_children, len(parents[0])))

    for n in range(num_children):
        child = [parents[g][i] for i,g in enumerate(random_gene_selection[n])]
        generation.append(child)

    return np.array(generation)
    
