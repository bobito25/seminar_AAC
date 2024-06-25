import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_simulated_annealing


def solve_tsp_from_file(filename: str = "TSP.txt") -> int:
    cities = []
    with open(filename, "r") as f:
        for line in f.readlines():
            city = line.split()
            cities.append((float(city[1]), float(city[2])))
    
    distance_matrix = euclidean_distance_matrix(cities)

    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

    return distance

print(solve_tsp_from_file())