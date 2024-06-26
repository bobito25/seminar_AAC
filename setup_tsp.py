import random

def setup_tsp(num_cities, max_x, max_y):
    cities = []
    for i in range(num_cities):
        cities.append((i+1, random.randint(0,max_x), random.randint(0,max_y)))
    return {"cities": cities}

def write_tsp_to_file(problem: dict):
    with open("TSP.txt", "w") as f:
        for city in problem["cities"]:
            f.write(f"{city[0]} {city[1]} {city[2]}\n")

TSP = setup_tsp(30, 100, 100)
write_tsp_to_file(TSP)