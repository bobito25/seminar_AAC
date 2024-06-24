import random
import math

class tester():

    def __init__(self):
        self.population = []
        self.parents = [[0,[1,2,3,4]],[0,[4,3,2,1]]]
        self.new_gen = []
    
    # Partially Matched Crossover
    def crossover_pmx(self):
        self.new_gen = []
        print("pairs:",[(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]])
        for parent1, parent2 in [(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]]:
            size = len(parent1)
    
            # Choose two random crossover points
            cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
            
            print("points:",cx_point1, cx_point2)
            
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
                used = set(offspring[cx_point1:cx_point2])
                for i in range(size):
                    if i < cx_point1 or i >= cx_point2:
                        if offspring[i] in used:
                            for item in parent:
                                if item not in used:
                                    offspring[i] = item
                                    used.add(item)
                                    break
                        else:
                            used.add(offspring[i])
                return offspring

            offspring1 = fix_duplicates(offspring1, parent2)
            offspring2 = fix_duplicates(offspring2, parent1)

            if offspring1.sort() != parent1.sort() or offspring2.sort() != parent1.sort():
                raise ValueError("not perm")

            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)
    
    def cycle_crossover(self):
        self.new_gen = []
        #print("pairs:",[(a[1], b[1]) for idx, a in enumerate(self.parents) for b in self.parents[idx + 1:]])
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

            #if offspring1.sort() != parent1.sort() or offspring2.sort() != parent1.sort():
            #    raise ValueError("not perm")
            
            self.new_gen.append(offspring1)
            self.new_gen.append(offspring2)



print("testing")

t = tester()



def cycle_crossover(parent1, parent2):
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
    
    return offspring1, offspring2

# Example usage:
parent1 = [1, 2, 3, 4, 5, 6, 7]
parent2 = [7, 4, 2, 1, 5, 6, 3]

offspring = cycle_crossover(parent1, parent2)
print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("Offspring:", offspring)

for i in range(1000):
    random.shuffle(t.parents[0][1])
    random.shuffle(t.parents[1][1])
    t.cycle_crossover()
    print("Offspring:", t.new_gen)

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