import random

from genetic_tsp.tsp import selectPopulation, geneticAlgorithm, drawMap

def setup_tsp(num_cities, max_x, max_y):
    cities = []
    for i in range(num_cities):
        cities.append((i+1, random.randint(0,max_x), random.randint(0,max_y)))
    return {"cities": cities}

def write_tsp_to_file(problem: dict):
    with open("genetic_tsp/TSP.txt", "w") as f:
        for city in problem["cities"]:
            f.write(f"{city[0]} {city[1]} {city[2]}\n")

def solve_tsp(problem: dict, params: dict, print_output=False) -> int:
    population_size = params["population_size"]
    tournament_selection_size = params["tournament_selection_size"]
    mutation_rate = params["mutation_rate"]
    crossover_rate = params["crossover_rate"]
    target = params["target"]
    generation_num = params["generation_num"]

    cities = problem["cities"]
    firstPopulation, firstFitest = selectPopulation(cities, population_size)
    answer, genNumber = geneticAlgorithm(
        firstPopulation,
        len(cities),
        tournament_selection_size,
        mutation_rate,
        crossover_rate,
        target,
        generation_num
    )
    
    if print_output:
        print("\n----------------------------------------------------------------")
        print("Generation: " + str(genNumber))
        print("Fittest chromosome distance before training: " + str(firstFitest[0]))
        print("Fittest chromosome distance after training: " + str(answer[0]))
        print("Target distance: " + str(TARGET))
        print("----------------------------------------------------------------\n")

        drawMap(cities, answer)

    return answer[0]

def solve_tsp_from_file(params: dict, filename: str = "TSP.txt") -> int:
    with open(filename, "r") as f:
        cities = []
        for line in f.readlines():
            city = line.split()
            cities.append((city[0], float(city[1]), float(city[2])))
        problem = {"cities": cities}
    return solve_tsp(problem, params)

POPULATION_SIZE = 5000
TOURNAMENT_SELECTION_SIZE = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
TARGET = 100.0
GENERATION_NUM = 30

#TSP = setup_tsp(5, 10, 10)
#write_tsp_to_file(TSP)
answer = solve_tsp_from_file({"population_size": POPULATION_SIZE, "tournament_selection_size": TOURNAMENT_SELECTION_SIZE, "mutation_rate": MUTATION_RATE, "crossover_rate": CROSSOVER_RATE, "target": TARGET, "generation_num": GENERATION_NUM})
print(answer)