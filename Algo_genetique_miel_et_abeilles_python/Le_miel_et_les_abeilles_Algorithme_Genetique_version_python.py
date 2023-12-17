#!/usr/bin/env python
# coding: utf-8

# # Algorithme génétique - Le miel et les abeilles

# In[487]:


# Importation des librairies
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


# In[489]:


df = pd.read_csv('abeilles')
# Ajouter la position de la ruche
position_ruche = pd.DataFrame({'x': [500], 'y': [500]})
df = pd.concat([position_ruche, df], ignore_index=True)


# In[500]:


df.head()


# # Création de la classe

# In[490]:


import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:
    unique_id = 0

    def __init__(self, population_size, df, method='top_parents', crossover_method='crossover', mutation_method='inversion', seed=42):
        self.population_size = population_size
        self.df = df
        self.seed = seed
        self.abeille_counter = 0  # Compteur d'abeilles pour générer des IDs uniques
        self.parcours_abeilles = self.initialize_population()
        self.dist_mat = self.calculate_distance_matrix()
        self.method = method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parents_ids = {}  # Dictionnaire pour répertorier les IDs des parents

    @classmethod
    def generate_unique_id(cls):
        cls.unique_id += 1
        return cls.unique_id

    def initialize_population(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        parcours_abeilles = []
        for _ in range(self.population_size):
            parcours = list(range(1, len(self.df)))  # Commence à partir de 1
            np.random.shuffle(parcours)
            if 0 not in parcours:
                parcours = [0] + parcours  # Ajoute 0 au début du parcours
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
            else:
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
        return parcours_abeilles
    
    def get_parents_ids(self):
        return self.parents_ids
    
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
        point1, point2 = 1, 49  # Indices de coupure arbitraires
        child1 = [0] + parent1[point1:point2]
        child2 = [0] + parent2[point1:point2]

        remaining_parent1 = [gene for gene in parent2 if gene not in child1]
        remaining_parent2 = [gene for gene in parent1 if gene not in child2]

        child1 += remaining_parent1
        child2 += remaining_parent2

        return child1, child2

    def crossover(self, parent1, parent2):
        start = np.random.randint(2, len(parent1))  # Démarre à partir de 2 pour exclure la ruche
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
        idx1, idx2 = np.random.choice(range(1, len(order)), 2, replace=False)  # Commence à partir de 1
        order[idx1], order[idx2] = order[idx2], order[idx1]
        return order

    def mutate_inversion(self, order):
        start, end = np.random.choice(range(1, len(order)), 2, replace=False)  # Commence à partir de 1
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

        best_fitness = float('inf')  # Initialise à l'infini
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

                # Répertorie les IDs des parents
                self.parents_ids[child1[0]] = (parent1[0], parent2[0])
                self.parents_ids[child2[0]] = (parent1[0], parent2[0])

                # Incrémente le compteur d'abeilles
                self.abeille_counter += 1

            self.parcours_abeilles = new_parcours_abeilles

            best_parcours_info = min(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
            best_parcours_id, best_parcours = best_parcours_info[0], best_parcours_info[1]
            best_distance = self.total_distance(best_parcours)
            fitness_history.append(best_distance)

            # Vérifie la stagnation
            if best_distance >= best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fitness = best_distance

            # Applique les mutations seulement en cas de stagnation
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

        return best_distance, best_parcours, np.mean(iteration_times),best_parcours_info  # Retourne également le temps moyen d'itération


# In[501]:


matrice[:5]


# In[502]:


plt.scatter(df['x'],df['y'])
plt.scatter(500,500,color='red')
plt.show()


# In[503]:


#Meilleur parcours de l'abeille de la première génération
parcours_meilleur_fitness = parcours_abeilles[index_meilleur_fitness]
chemin_meilleur_fitness_x = [df.loc[p]['x'] for p in parcours_meilleur_fitness]
chemin_meilleur_fitness_y = [df.loc[p]['y'] for p in parcours_meilleur_fitness]

# Afficher le parcours avec Matplotlib
plt.plot(chemin_meilleur_fitness_x, chemin_meilleur_fitness_y, marker='o', linestyle='-', color='blue')
plt.scatter(df['x'], df['y'], color='red', marker='o', label='Fleurs')
plt.scatter(df.loc[0]['x'], df.loc[0]['y'], color='green', marker='s', label='Ruche')

plt.title('Parcours avec le Meilleur Fitness Score')
plt.xlabel('Coordonnée X')
plt.ylabel('Coordonnée Y')
plt.legend()
plt.show()


# In[504]:


# Créer un DataFrame avec l'identifiant de l'abeille et son score fitness pour la première abeille
resultats_df = pd.DataFrame({'Abeille': range(1, nombre_abeilles + 1), 'Fitness': distances_abeilles,'parcours':parcours_abeilles})

# Afficher les dix premières lignes du DataFrame
resultats_df.sort_values('Fitness',ascending=True)[:10]


# ## Comparaison de différentes méthodes

# In[505]:


# Initialiser le DataFrame pour stocker les résultats
all_results_df = pd.DataFrame(columns=['Method', 'Crossover Method', 'Mutation Method', 'Fitness Score', 'Best Parcours','Time per Generation'])

# Boucle sur toutes les combinaisons possibles
methods = ['top_parents']
crossover_methods = ['crossover', 'crossover_two_points']
mutation_methods = ['swap', 'inversion']

results_list = []

# Boucle sur toutes les combinaisons de méthodes
for method in methods:
    for crossover_method in crossover_methods:
        for mutation_method in mutation_methods:
        # Initialiser l'algorithme génétique avec les méthodes actuelles et la graine par défaut=42
        # Créer une instance de la classe GeneticAlgorithm
        # Imprimer les informations de la méthode sélectionnée
            print(f"Method: {method}, Crossover Method: {crossover_method}, Mutation Method: {mutation_method}")

            instance_genetic_algorithm = GeneticAlgorithm(population_size=100, df=df)

            # Exécutez l'algorithme génétique
            best_distance, best_parcours, mean_iteration_time,best_parcours_info = instance_genetic_algorithm.run_genetic_algorithm(generations=500, mutation_rate=0.04)

            # Add results to the list
            results_list.append({
                'Method': method,
                'Crossover Method': crossover_method,
                 'Mutation Method': mutation_method,
                'Fitness Score': best_distance,
                'Best Parcours': best_parcours,
                'Time per Generation': mean_iteration_time
                })

# Create the DataFrame from the list
results_df = pd.DataFrame(results_list)


# In[492]:


results_df.sort_values('Fitness Score')

# On remarque que la méthode de mutations 'Swap' et celle de sélection 'Roulette' ne permettent pas d'optimiser le Fitness Score. Je les retire et recommance le test
# # Comparaison des méthodes retenues

# In[506]:


# C'est une copie du premier résultat
results_df_2 = results_df
results_df_2.sort_values('Fitness Score')


# # Test 2: Modification du taux de mutation

# In[507]:


instance_genetic_algorithm = GeneticAlgorithm(population_size=100, df=df, method='top_parents', crossover_method='crossover', mutation_method='inversion', seed=42)
best_distance, best_parcours,mean_iteration_time,best_parcours_info = instance_genetic_algorithm.run_genetic_algorithm(generations=300, mutation_rate=0.04)


# In[508]:


# Obtention du dictionnaire des IDs des parents
parents_ids = instance_genetic_algorithm.get_parents_ids()
print(best_parcours_info)


# In[509]:


import matplotlib.pyplot as plt
import networkx as nx

def hierarchy_pos(G, root=None, width=1., vert_gap=0.3, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap * 2, xcenter=nextx, pos=pos,
                                 parent=root, parsed=parsed)

    return pos


def plot_genealogy(parents_ids, target_id, generations=5):
    G = nx.DiGraph()

    def add_parents_to_graph(graph, current_id, generations_left):
        if generations_left <= 0:
            return

        parents = parents_ids.get(current_id)
        if parents:
            parent1, parent2 = parents
            graph.add_edge(current_id, parent1)
            graph.add_edge(current_id, parent2)

            # Récursivement ajouter les parents
            add_parents_to_graph(graph, parent1, generations_left - 1)
            add_parents_to_graph(graph, parent2, generations_left - 1)

    add_parents_to_graph(G, target_id, generations)

    pos = hierarchy_pos(G, target_id, vert_gap=0.8)
    plt.figure(figsize=(12, 9))  # Ajustez les valeurs de largeur et hauteur selon les besoins
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=600, node_color='skyblue', font_size=8, arrowsize=15)
    plt.title(f'Généalogie de l\'abeille avec ID {target_id} sur {generations} générations')
    plt.show()


# Utilisation de la meilleure abeille obtenue (faut s'assurer d'avoir exécuté l'algorithme au moins 10 générations)
best_abeille_id = 14950
plot_genealogy(parents_ids, best_abeille_id, generations=5)


# In[512]:


import numpy as np
import matplotlib.pyplot as plt

# Fonction pour calculer la fitness d'un individu
def calculate_fitness(individual):
    # Ici, on peut définir la fonction de fitness en fonction des caractéristiques de l'individu
    # Plus la valeur est basse, meilleure est la fitness (adaptation)
    return np.sum(individual)

# Fonction de mutation
def mutate(individual):
    mutated_index = np.random.randint(len(individual))
    individual[mutated_index] = 1 - individual[mutated_index]  # Inversion de 0 à 1 ou de 1 à 0
    return individual

# Fonction de croisement (crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Algorithme génétique principal
def genetic_algorithm(population_size, individual_size, generations):
    population = np.random.randint(2, size=(population_size, individual_size))  # Population initiale

    fitness_history = []  # Pour stocker les scores de fitness à chaque génération

    for generation in range(generations):
        # Calcul du fitness de chaque individu dans la population
        fitness_values = np.array([calculate_fitness(individual) for individual in population])

        # Sélection des parents avec la méthode de roulette
        probabilities = fitness_values / np.sum(fitness_values)
        parents_indices = np.random.choice(range(population_size), size=population_size // 2, p=probabilities)
        parents = population[parents_indices]

        new_population = []

        # Croisement et mutation pour créer de nouveaux individus
        for _ in range(population_size // 2):
            parent1 = parents[np.random.choice(range(population_size // 2))]
            parent2 = parents[np.random.choice(range(population_size // 2))]

            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            if np.random.rand() < mutation_rate:
                child1 = mutate(child1)
            if np.random.rand() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = np.array(new_population)

        # Calcul du fitness du meilleur individu de la génération actuelle
        best_individual = population[np.argmin(fitness_values)]
        best_fitness = calculate_fitness(best_individual)
        fitness_history.append(best_fitness)

    # Affichage des résultats en dehors de la boucle de génération
    print("Meilleur individu trouvé :", best_individual)
    print("Fitness :", best_fitness)

    # Affichage du graphique
    plt.plot(fitness_history)
    plt.title('Evolution du fitness par génération')
    plt.xlabel('Génération')
    plt.ylabel('Fitness Score')
    plt.show()

    return best_individual, best_fitness

if __name__ == "__main__":
    population_size = 50
    individual_size = 10
    generations = 100
    mutation_rate = 0.1

    best_individual, best_fitness = genetic_algorithm(population_size, individual_size, generations)


# In[ ]:




