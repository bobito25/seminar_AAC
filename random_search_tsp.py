import numpy as np
import random

def random_search(problem,n):
    cities = problem["cities"]
    best = np.inf
    for i in range(n):
        random.shuffle(cities)
        dist = calcDistance(cities)
        if dist < best:
            best = dist
    return best


def random_search_tsp_from_file(filename: str = "TSP.txt") -> int:
    with open(filename, "r") as f:
        cities = []
        for line in f.readlines():
            city = line.split()
            cities.append((city[0], float(city[1]), float(city[2])))
        problem = {"cities": cities}
    return random_search(problem,500000)

# calculating distance of the cities -> fitness of this instance (subject to minimization)
def calcDistance(nodes):
    # Extract only the x and y coordinates, ignoring the node names
    path = np.array([node[1:] for node in nodes])
    
    # Calculate the differences between consecutive points
    diff = np.diff(path, axis=0)
    
    # Calculate the Euclidean distance for each step
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Sum up all the distances
    total_distance = np.sum(distances)
    
    return total_distance

print(random_search_tsp_from_file())