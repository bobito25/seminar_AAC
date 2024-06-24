import random
import math
import numpy as np
import time

from fitness.base_ff_classes.base_ff import base_ff


class genetic_tsp(base_ff):
    """
    Basic fitness function template for writing new fitness functions. This
    basic template inherits from the base fitness function class, which
    contains various checks and balances.
    
    Note that all fitness functions must be implemented as a class.
    
    Note that the class name must be the same as the file name.
    
    Important points to note about base fitness function class from which
    this template inherits:
    
      - Default Fitness values (can be referenced as "self.default_fitness")
        are set to NaN in the base class. While this can be over-written,
        PonyGE2 works best when it can filter solutions by NaN values.
    
      - The standard fitness objective of the base fitness function class is
        to minimise fitness. If the objective is to maximise fitness,
        this can be over-written by setting the flag "maximise = True".
    
    """

    # The base fitness function class is set up to minimise fitness.
    # However, if you wish to maximise fitness values, you only need to
    # change the "maximise" attribute here to True rather than False.
    # Note that if fitness is being minimised, it is not necessary to
    # re-define/overwrite the maximise attribute here, as it already exists
    # in the base fitness function class.
    maximise = False

    cross_over_timeout = 0.1 # in seconds

    def __init__(self):
        """
        All fitness functions which inherit from the bass fitness function
        class must initialise the base class during their own initialisation.
        """
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
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        """
        Default fitness execution call for all fitness functions. When
        implementing a new fitness function, this is where code should be added
        to evaluate target phenotypes.
        
        There is no need to implement a __call__() method for new fitness
        functions which inherit from the base class; the "evaluate()" function
        provided here allows for this. Implementing a __call__() method for new
        fitness functions will over-write the __call__() method in the base
        class, removing much of the functionality and use of the base class.
                
        :param ind: An individual to be evaluated.
        :param kwargs: Optional extra arguments.
        :return: The fitness of the evaluated individual.
        """

        # Evaluate the fitness of the phenotype
        fitness = self.parse_genome(ind.phenotype)
        return fitness
    
    def parse_genome(self, genome):
        action_list = genome.split(" ")
        while len(action_list) != 0:
            next_action = action_list.pop(0)
            print(next_action)
            action = self.actions[next_action]
            if action["num_params"] != 0:
                params = [action_list.pop(0) for _ in range(action["num_params"])]
                print("params:",params)
                action["f"](*params)
            else:
                action["f"]()
            if not self.population:
                return np.NaN
            if len(self.population) > 1000:
                print("population limit reached, pruning...")
                self.population = self.population[:1000]
            self.validate()
            print("population size:",len(self.population))
            print("parent size:",len(self.parents))
            print("new gen size:",len(self.new_gen))
        return self.get_fitness()
    
    def get_fitness(self):
        return min(self.population, key=lambda x: x[0])[0]

    def validate(self):
        for i in self.population:
            if not i:
                raise ValueError("member is falsey:",i)
            if not i[0] or not i[1]:
                raise ValueError("attribute of member is falsey:",i)
            try: 
                d = calcDistance(i[1])
            except Exception as e:
                raise ValueError("error occured in dist calc of member:",i) from e
            if i[0] != d:
                raise ValueError("wrong distance in member:", i)
            if list(i[1]).sort() != self.cities.sort():
                raise ValueError("member is not valid tsp solution:",i)
        for i in self.new_gen:
            if not i:
                raise ValueError("member of new gen is falsey:",i)
            if list(i).sort() != self.cities.sort():
                raise ValueError("member of new gen is not valid tsp solution:",i)
        for i in self.parents:
            if not i:
                raise ValueError("member of parents is falsey:",i)
            if not i[0] or not i[1]:
                raise ValueError("attribute of member of parents is falsey:",i)
            try: 
                d = calcDistance(i[1])
            except Exception as e:
                raise ValueError("error occured in dist calc of member of parents:",i) from e
            if i[0] != d:
                raise ValueError("wrong distance in member of parents:", i)
            if list(i[1]).sort() != self.cities.sort():
                raise ValueError("member of parents is not valid tsp solution:",i)

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
        if len(self.population) < tournament_size:
            best = min(self.population, key=lambda x: x[0])
            self.parents = [best]*int(len(self.population)*percent)
            return
        for _ in range(int(len(self.population)*percent)):
            tournament = random.sample(self.population, tournament_size)
            best = min(tournament, key=lambda x: x[0])
            self.parents.append(best)

    # Partially Matched Crossover
    def crossover_pmx(self):
        timeout = time.time() + self.cross_over_timeout
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
    
            # Choose two random crossover points
            cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
            
            # Initialize offspring
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            
            # Perform crossover
            for i in range(cx_point1, cx_point2):
                # Find the values to be swapped
                val1, val2 = parent1[i], parent2[i]
                
                # Find the positions of these values in the other parent
                pos1 = offspring1.index(val2)
                pos2 = offspring2.index(val1)
                
                # Swap the values
                offspring1[i], offspring1[pos1] = offspring1[pos1], offspring1[i]
                offspring2[i], offspring2[pos2] = offspring2[pos2], offspring2[i]
            
            # Verify and fix any remaining duplicates
            def fix_duplicates(offspring, parent):
                used = list(offspring[cx_point1:cx_point2])
                for i in range(size):
                    if i < cx_point1 or i >= cx_point2:
                        if offspring[i] in used:
                            for item in parent:
                                if item not in used:
                                    offspring[i] = item
                                    used.append(item)
                                    break
                        else:
                            used.append(offspring[i])
                return offspring

            offspring1 = fix_duplicates(offspring1, parent2)
            offspring2 = fix_duplicates(offspring2, parent1)

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)

            if time.time() > timeout:
                print("crossover timeout, continuing...")
                return
    
    # Order Crossover
    def crossover_ox(self):
        timeout = time.time() + self.cross_over_timeout
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
            
            # Choose two random crossover points
            cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
            
            def create_offspring(p1, p2):
                # Initialize offspring with a copy of the segment from p1
                offspring = [None] * size
                offspring[cx_point1:cx_point2] = p1[cx_point1:cx_point2]
                
                # Fill the remaining positions with values from p2
                remaining = [item for item in p2 if item not in offspring[cx_point1:cx_point2]]
                index = cx_point2
                for item in remaining:
                    if index == size:
                        index = 0
                    while offspring[index] is not None:
                        index = (index + 1) % size
                    offspring[index] = item
                    index = (index + 1) % size
                
                return offspring
            
            # Create two offspring
            offspring1 = create_offspring(parent1, parent2)
            offspring2 = create_offspring(parent2, parent1)

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)

            if time.time() > timeout:
                print("crossover timeout, continuing...")
                return

    # Cycle Crossover
    def crossover_cx(self):
        timeout = time.time() + self.cross_over_timeout
        self.new_gen = []
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
            
            def create_offspring(p1, p2):
                offspring = [None] * size
                index = 0
                
                while None in offspring:
                    if offspring[index] is None:
                        # Start of a new cycle
                        cycle = []
                        while True:
                            cycle.append(index)
                            offspring[index] = p1[index]
                            index = p1.index(p2[index])
                            if index == cycle[0]:
                                break
                        
                        # Fill the remaining positions in the cycle with values from p2
                        for i in range(size):
                            if i not in cycle and offspring[i] is None:
                                offspring[i] = p2[i]
                    
                    index = (index + 1) % size
                
                return offspring
            
            # Create two offspring
            offspring1 = create_offspring(parent1, parent2)
            offspring2 = create_offspring(parent2, parent1)

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)

            if time.time() > timeout:   
                print("crossover timeout, continuing...")
                return

    # Swap Mutation
    def mutate_swap(self, mut_rate):
        num = len(self.new_gen)
        mut_rate = float(mut_rate)
        for i in range(len(self.new_gen)):
            if random.random() < mut_rate:
                mutated = self.new_gen[i]
                idx1, idx2 = random.sample(range(len(mutated)), 2)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        if num != len(self.new_gen):
            raise ValueError("mutation changed number of members of new gen")

    # Inversion Mutation
    def mutate_inversion(self, mut_rate):
        num = len(self.new_gen)
        mut_rate = float(mut_rate)
        for i in range(len(self.new_gen)):
            if random.random() < mut_rate:
                mutated = self.new_gen[i]
                idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))
                mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])
        if num != len(self.new_gen):
            raise ValueError("mutation changed number of members of new gen")

    # Scramble Mutation
    def mutate_scramble(self, mut_rate):
        num = len(self.new_gen)
        mut_rate = float(mut_rate)
        for i in range(len(self.new_gen)):
            if random.random() < mut_rate:
                mutated = self.new_gen[i]
                idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))
                subset = mutated[idx1:idx2+1]
                random.shuffle(subset)
                mutated[idx1:idx2+1] = subset
        if num != len(self.new_gen):
            raise ValueError("mutation changed number of members of new gen")
    
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
        if not self.new_gen:
            self.population = []
            return
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
            raise ValueError("this should not be reachable",locals())

        self.population = [[calcDistance(ind), ind] for ind in [select_one() for _ in range(len(self.population))]]
        self.new_gen = []

    def replacement_random(self):
        if not self.new_gen:
            return
        to_replace = random.sample(range(len(self.population)), len(self.population))
        for i in to_replace:
            ind = random.choice(self.new_gen)
            self.population[i] = [calcDistance(ind), ind]
        self.new_gen = []

# get cities info
def getCity():
    cities = []
    f = open("TSP.txt")
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            (node_city_val[0], float(node_city_val[1]), float(node_city_val[2]))
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