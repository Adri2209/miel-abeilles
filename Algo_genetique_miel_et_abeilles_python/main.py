# Importation des librairies
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

df = pd.read_csv('abeilles')

# Ajouter la position de la ruche
position_ruche = pd.DataFrame({'x': [500], 'y': [500]})
df = pd.concat([position_ruche, df], ignore_index=True)

from beehive import BeeHive

# Définissez les paramètres
population_size = 100
nombre_fleurs = 50
parcours_abeilles = []

# Creation d'une instancve BeeHive
bee_hive = BeeHive(population_size, df)

# ExExecution de l'algorithme génétique
result = bee_hive.run_genetic_algorithm(generations=800, mutation_rate=0.04)

# Accéss aux résultats
best_distance, best_parcours, mean_iteration_time, best_parcours_info = result



