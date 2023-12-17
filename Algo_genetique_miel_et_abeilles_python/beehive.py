import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

class BeeHive:
    unique_id = 0

    def __init__(self, population_size, df, method='top_parents', crossover_method='crossover', mutation_method='inversion', seed=42):
        self.population_size = population_size
        self.df = df
        self.seed = seed
        self.abeille_counter = 0
        self.parcours_abeilles = self.initialize_population()
        self.dist_mat = self.calculate_distance_matrix()
        self.method = method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parents_ids = {}

    @classmethod
    def generate_unique_id(cls):
        cls.unique_id += 1
        return cls.unique_id

    def initialize_population(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        parcours_abeilles = []
        for _ in range(self.population_size):
            parcours = list(range(1, len(self.df)))
            np.random.shuffle(parcours)
            if 0 not in parcours:
                parcours = [0] + parcours
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
            else:
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
        return parcours_abeilles

    def ajuster_parcours(self, parcours):
        position_ruche = parcours.index(0)
        return parcours[position_ruche:] + parcours[:position_ruche]

    def total_distance(self, order):
        distances = [self.dist_mat[order[i], order[(i + 1) % len(order)]] for i in range(len(order))]
        return sum(distances)

    def calculate_distance_matrix(self):
        A = np.array(self.df)
        B = A.copy()
        return distance_matrix(A, B, p=1)

    def crossover_two_points(self, parent1, parent2):
        point1, point2 = 1, 49
        child1 = [0] + parent1[point1:point2]
        child2 = [0] + parent2[point1:point2]

        remaining_parent1 = [gene for gene in parent2 if gene not in child1]
        remaining_parent2 = [gene for gene in parent1 if gene not in child2]

        child1 += remaining_parent1
        child2 += remaining_parent2

        return child1, child2

    def crossover(self, parent1, parent2):
        start = np.random.randint(2, len(parent1))
        end = np.random.randint(start, len(parent1))
        child = [0] + [None] * (len(parent1) - 1)
        for i in range(start, end):
            child[i] = parent1[i]
        idx = 1
        for i in range(len(parent2)):
            if parent2[i] not in child:
                while child[idx] is not None:
                    idx += 1
                child[idx] = parent2[i]
        return child

    def top_parents_selection(self, num_parents=50):
        sorted_population = sorted(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
        return sorted_population[:num_parents]

    def mutate_swap(self, order):
        idx1, idx2 = np.random.choice(range(1, len(order)), 2, replace=False)
        order[idx1], order[idx2] = order[idx2], order[idx1]
        return order

    def mutate_inversion(self, order):
        start, end = np.random.choice(range(1, len(order)), 2, replace=False)
        order[start:end + 1] = order[start:end + 1][::-1]
        return order

    def ensure_zero_first(self, order):
        zero_index = order.index(0)
        order[0], order[zero_index] = order[zero_index], order[0]
        return order

    def get_parents_ids(self):
        return self.parents_ids

    def run_genetic_algorithm(self, generations=800, mutation_rate=0.04):
        fitness_history = []
        iteration_times = []

        best_fitness = float('inf')
        stagnation_counter = 0

        for generation in range(generations):
            start_time = time.time()
            self.parcours_abeilles = sorted(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
            new_parcours_abeilles = []

            for i in range(len(self.parcours_abeilles) // 2):
                parent1 = self.parcours_abeilles[i]
                parent2 = self.parcours_abeilles[i + 1]

                if self.method == "top_parents":
                    parents = self.top_parents_selection(2)
                    parent1, parent2 = parents[0], parents[1]
                else:
                    raise ValueError("Méthode de sélection des parents non valide")

                if self.crossover_method == "crossover":
                    child1 = (self.abeille_counter, self.crossover(parent1[1], parent2[1]))
                    child2 = (self.abeille_counter, self.crossover(parent2[1], parent1[1]))
                elif self.crossover_method == "crossover_two_points":
                    child1, child2 = self.crossover_two_points(parent1[1], parent2[1])
                else:
                    raise ValueError("Méthode de croisement non valide")

                if np.random.rand() < mutation_rate:
                    if self.mutation_method == "swap":
                        child1 = (self.abeille_counter, self.mutate_swap(child1[1]))
                    elif self.mutation_method == "inversion":
                        child1 = (self.abeille_counter, self.mutate_inversion(child1[1]))
                    else:
                        raise ValueError("Méthode de mutation non valide")

                if np.random.rand() < mutation_rate:
                    if self.mutation_method == "swap":
                        child2 = (self.abeille_counter, self.mutate_swap(child2[1]))
                    elif self.mutation_method == "inversion":
                        child2 = (self.abeille_counter, self.mutate_inversion(child2[1]))
                    else:
                        raise ValueError("Méthode de mutation non valide")

                new_parcours_abeilles.extend([child1, child2])

                self.parents_ids[child1[0]] = (parent1[0], parent2[0])
                self.parents_ids[child2[0]] = (parent1[0], parent2[0])

                self.abeille_counter += 1

            self.parcours_abeilles = new_parcours_abeilles

            best_parcours_info = min(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
            best_parcours_id, best_parcours = best_parcours_info[0], best_parcours_info[1]
            best_distance = self.total_distance(best_parcours)
            fitness_history.append(best_distance)

            if best_distance >= best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fitness = best_distance

            if stagnation_counter > 50:
                for abeille_info in self.parcours_abeilles:
                    abeille_id, parcours = abeille_info[0], abeille_info[1]
                    if np.random.rand() < mutation_rate:
                        mutated_parcours = None
                        if self.mutation_method == "swap":
                            mutated_parcours = self.mutate_swap(parcours)
                        elif self.mutation_method == "inversion":
                            mutated_parcours = self.mutate_inversion(parcours)
                
                        if mutated_parcours is not None:
                            self.parcours_abeilles.append((self.abeille_counter, mutated_parcours))
                            self.abeille_counter += 1

            end_time = time.time()
            iteration_time = end_time - start_time
            iteration_times.append(iteration_time)

        plt.plot(fitness_history)
        plt.title(f'Evolution du fitness par génération\nSelection: {self.method}, Crossover: {self.crossover_method}, Mutation: {self.mutation_method}, Mutation Rate: {mutation_rate}')
        plt.xlabel('Génération')
        plt.ylabel('Fitness Score')
        plt.show()

        print("Meilleur parcours trouvé :", best_parcours)
        print("Distance totale :", best_distance)
        print("Temps moyen par génération :", np.mean(iteration_times))

        return best_distance, best_parcours, np.mean(iteration_times), best_parcours_info

bee_hive_instance = BeeHive(population_size, df)
bee_hive_instance.run_genetic_algorithm(generations=800, mutation_rate=0.04)
