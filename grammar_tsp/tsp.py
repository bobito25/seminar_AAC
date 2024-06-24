import random
import math

# get cities info
def getCity():
    cities = []
    f = open("TSP.txt")
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [node_city_val[0], float(node_city_val[1]), float(node_city_val[2])]
        )

    return cities

# calculating distance of the cities -> fitness of this instance (subject to minimization)
def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]

        d = math.sqrt(
            math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2)
        )

        total_sum += d

    cityA = cities[0]
    cityB = cities[-1]
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))

    total_sum += d

    return total_sum

class parser():
    def __init__(self):
        self.cities = getCity()
        self.actions = {
            "init_pop": {"f": self.init_population, "num_params": 1},
            "roulette": {"f": self.select_parents_roulette, "num_params": 1},
            "tournament": {"f": self.select_parents_tournament, "num_params": 2},
            "rank": {"f": self.select_parents_rank, "num_params": 1},
            "pmx": {"f": self.crossover_pmx, "num_params": 0},
            "ox": {"f": self.crossover_ox, "num_params": 0},
            "cx": {"f": self.crossover_cx, "num_params": 0},
            "swap": {"f": self.mutate_swap, "num_params": 1},
            "inversion": {"f": self.mutate_inversion, "num_params": 1},
            "scramble": {"f": self.mutate_scramble, "num_params": 1},
            "generational": {"f": self.replacement_generational, "num_params": 0},
            "steady_state": {"f": self.replacement_steady_state, "num_params": 1},
            "elitism": {"f": self.replacement_elitism, "num_params": 1},
            "mu_plus_lambda": {"f": self.replacement_mu_plus_lambda_, "num_params": 0},
            "mu_comma_lambda": {"f": self.replacement_mu_comma_lambda, "num_params": 0},
            "fitness_proportionate": {"f": self.replacement_fitness_proportionate, "num_params": 0},
            "random": {"f": self.replacement_random, "num_params": 0}
        }
        self.population = []
        self.parents = []
        self.new_gen = []

    def parse(self, genome):
        action_list = genome.split(" ")
        while len(action_list) != 0:
            next_action = action_list.pop(0)
            action = self.actions[next_action]
            if action["num_params"] != 0:
                params = [action_list.pop(0) for _ in range(action["num_params"])]
                action["f"](*params)
            else:
                action["f"]()
        return self.get_fitness()
    
    def get_fitness(self):
        return min(self.population, key=lambda x: x[0])

    # init the population
    def init_population(self, size):
        size = int(size)
        population = []

        for i in range(size):
            c = self.cities.copy()
            random.shuffle(c)
            distance = calcDistance(c)
            population.append([distance, c])
        
        self.population = population

    def select_parents_rank(self, percent):
        percent = float(percent)
        sorted_population = sorted(self.population, key=lambda x: x[0])
        num_parents = int(len(self.population) * percent)
        self.parents = sorted_population[:num_parents]

    def select_parents_roulette(self, percent):
        percent = float(percent)
        total_fitness = sum(1 / ind[0] for ind in self.population)  # Inverse of distance for fitness
        selection_probs = [(1 / ind[0]) / total_fitness for ind in self.population]
        
        cumulative_probs = []
        cumulative_sum = 0
        for prob in selection_probs:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)

        def select_one():
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    return self.population[i]

        self.parents = [select_one() for _ in range(int(len(self.population)*percent))]

    def select_parents_tournament(self, tournament_size, percent):
        tournament_size = int(tournament_size)
        percent = float(percent)
        self.parents = []
        for _ in range(int(len(self.population)*percent)):
            tournament = random.sample(self.population, tournament_size)
            best = min(tournament, key=lambda x: x[0])
            self.parents.append(best)

    # Partially Matched Crossover
    def crossover_pmx(self):
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
            offspring1, offspring2 = [-1]*size, [-1]*size

            # Choose crossover points
            point1, point2 = sorted(random.sample(range(size), 2))

            # Apply crossover between cxpoints
            offspring1[point1:point2+1] = parent1[point1:point2+1]
            offspring2[point1:point2+1] = parent2[point1:point2+1]

            def pmx_fill(offspring, parent):
                for i in range(point1, point2+1):
                    if parent[i] not in offspring:
                        pos = i
                        while point1 <= pos <= point2:
                            pos = parent.index(parent1[pos])
                        offspring[pos] = parent[i]

                for i in range(size):
                    if offspring[i] == -1:
                        offspring[i] = parent[i]

            pmx_fill(offspring1, parent2)
            pmx_fill(offspring2, parent1)

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)
    
    # Order Crossover
    def crossover_ox(self):
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
            offspring1, offspring2 = [-1]*size, [-1]*size

            # Choose crossover points
            point1, point2 = sorted(random.sample(range(size), 2))

            # Apply crossover between cxpoints
            offspring1[point1:point2+1] = parent1[point1:point2+1]
            offspring2[point1:point2+1] = parent2[point1:point2+1]

            def ox_fill(offspring, parent):
                parent_index = point2 + 1
                for i in range(size):
                    offspring_index = (point2 + 1 + i) % size
                    while offspring[offspring_index] != -1:
                        offspring_index = (offspring_index + 1) % size
                    while parent[parent_index % size] in offspring:
                        parent_index += 1
                    offspring[offspring_index] = parent[parent_index % size]

            ox_fill(offspring1, parent2)
            ox_fill(offspring2, parent1)

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)

    # Cycle Crossover
    def crossover_cx(self):
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
            offspring1, offspring2 = [-1]*size, [-1]*size
            cycles = [0] * size

            cycle = 1
            for i in range(size):
                if cycles[i] == 0:
                    x = i
                    while cycles[x] == 0:
                        cycles[x] = cycle
                        x = parent1.index(parent2[x])
                    cycle += 1

            for i in range(size):
                if cycles[i] % 2 == 1:
                    offspring1[i] = parent1[i]
                    offspring2[i] = parent2[i]
                else:
                    offspring1[i] = parent2[i]
                    offspring2[i] = parent1[i]

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)

    # Swap Mutation
    def mutate_swap(self, mut_rate):
        mut_rate = float(mut_rate)
        for i in range(len(self.population)):
            if random.random() < mut_rate:
                individual = self.population[i][1]
                size = len(individual)
                pos1, pos2 = random.sample(range(size), 2)
                individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

    # Inversion Mutation
    def mutate_inversion(self, mut_rate):
        mut_rate = float(mut_rate)
        for i in range(len(self.population)):
            if random.random() < mut_rate:
                individual = self.population[i][1]
                size = len(individual)
                pos1, pos2 = sorted(random.sample(range(size), 2))
                individual[pos1:pos2+1] = reversed(individual[pos1:pos2+1])
                return individual

    # Scramble Mutation
    def mutate_scramble(self, mut_rate):
        mut_rate = float(mut_rate)
        for i in range(len(self.population)):
            if random.random() < mut_rate:
                individual = self.population[i][1]
                size = len(individual)
                pos1, pos2 = sorted(random.sample(range(size), 2))
                subset = individual[pos1:pos2+1]
                random.shuffle(subset)
                individual[pos1:pos2+1] = subset
    
    def replacement_generational(self):
        self.population = sorted([[calcDistance(ind), ind] for ind in self.new_gen], key=lambda x: x[0])
        self.new_gen = []

    def replacement_steady_state(self, percent):
        percent = float(percent)
        len_before = len(self.population)
        num = int(len(self.population) * percent)
        self.population = sorted(self.population, key=lambda x: x[0])[:num]
        new_sorted = sorted([[calcDistance(ind), ind] for ind in self.new_gen], key=lambda x: x[0])
        self.population += new_sorted[:len_before-num]
        self.new_gen = []

    def replacement_elitism(self, percent):
        percent = float(percent)
        elite_size = int(len(self.population) * percent)
        elite = sorted(self.population, key=lambda x: x[0])[:elite_size]
        remaining_offspring = sorted([[calcDistance(ind), ind] for ind in self.new_gen], key=lambda x: x[0])[:len(self.population) - elite_size]
        self.population = sorted(elite + remaining_offspring, key=lambda x: x[0])
        self.new_gen = []

    def replacement_mu_plus_lambda_(self):
        combined = self.population + [[calcDistance(ind), ind] for ind in self.new_gen]
        self.population = sorted(combined, key=lambda x: x[0])[:len(self.population)]
        self.new_gen = []

    def replacement_mu_comma_lambda(self):
        self.population = sorted([[calcDistance(ind), ind] for ind in self.new_gen], key=lambda x: x[0])[:len(self.population)]
        self.new_gen = []

    def replacement_fitness_proportionate(self):
        total_fitness = sum(1 / calcDistance(ind) for ind in self.new_gen)  # Inverse of distance for fitness
        selection_probs = [(1 / calcDistance(ind)) / total_fitness for ind in self.new_gen]
        
        cumulative_probs = []
        cumulative_sum = 0
        for prob in selection_probs:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)

        def select_one():
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    return self.new_gen[i]

        self.population = [[calcDistance(ind), ind] for ind in [select_one() for _ in range(len(self.population))]]
        self.new_gen = []

    def replacement_random(self):
        to_replace = random.sample(range(self.population), len(self.population))
        for i in to_replace:
            ind = random.choice(self.new_gen)
            self.population[i] = [calcDistance(ind), ind]
        self.new_gen = []