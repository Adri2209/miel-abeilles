# Projet Le miel et les abeilles - Algorithme génétique

## Problématique

 ### Nous avons une ruche avec 100 abeilles plus la reine,l'objectif est de trouver le parcours le plus court de chaque abeille qui visite toutes les fleurs une fois et reviens à la ruche qui est son point de départ. 
 ### C'est le même principe que pour le voyageur de commerce (TSP)
 
 ![image.png](attachment:image.png)

 ## Solution (Approche utilisé)

 ### J'ai d'abord observé la ruche et les abeilles puis fait le parcours d'une seule abeille afin d'observer son comportement, ensuite j'ai crée la classe de mon algorithme génétique pour un trajet aleatoire.

### Ma classe est initialisé avec des paramètres pour la taille de la populations des abeilles, le dataframe avec les coordonnées des fleurs ainsi que les méthodes de sélection, de croisement et de mutation des abeilles.

### La matrice de distance est calculée à partir des coordonnées du dataframe, la selection des parents, le croisement et la mutation se font dans le but de faire évoluer la population vers des meilleurs résultats. 

### Les id des parents sont enregistrés, et l'algorithme génétique est éxecuté sur plusieurs générations si les résultats stagnent il y a des mutations supplementaires qui sont apliquées.

### Nous pouvons observer les fitness scores des génération car elles sont affichés, ainsi que le meilleur fitness score.

### A la fin de l'execution du meilleur parcours, la distance totale et le temps moyen sont affiché pour chaque génération.

### Afin d'optimiser les résultat j'ai fait appel à plusieurs méthodes comme la fonction d'accès aux données (Getter), Fitness Evaluation Function, Data Manipulation et l'opération de recombinaison génétique (crossover_two_points) qui est une méthode de  croisement utilisant deux point de coupure que j'ai fixé à 1 et 49(indices arbritaires) 

## Organisation du code

### Mon code est organisé de la façon suivante:

### - Impotation des bibliotèques Python
### - Importation du fichier .csv
### -  Création de la classe avec tous les attributs
### - Initialisation de la population avec un parcours aleatoire puis l'utilisation de la matrice de distance pour évaluer la performace des parcours
### - Les opérateurs Génétiques : corssover et crossover_two_points; méthodes de mutation swap et inversion; méthode pour ajuster le parcours en verifiant que la ruche reste à la première position
### - La méthode de selection des parents basé sur la distance totale du parcours
### - L'évolution de la population sur un certain nombre de génération et la surveillance de la stagnation afin d'introduire des mutations supplementaires après une période sans amélioration significative
### - Graphique de l'évolution du fitness au fil des générations et affichage du meilleur parcours trouvé
### - La création d'un arbre généalogique de la meilleure abeille sur 5 générations

## Execution du code

### Le dataframe

![image.png](attachment:image.png)

### La ruche et les abeilles


![image-2.png](attachment:image-2.png)

### Comparaison de differentes méthodes

![image-7.png](attachment:image-7.png)
![image-5.png](attachment:image-5.png)

### Modification du taux de mutation

![image-6.png](attachment:image-6.png)

## Résultats

### Arbre généalogique de la meilleure abeille sur 5 génération avec 

![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)

## Conclusion

### L'algorithme est conçu de manière modulaire ce qui permet aux utilisateurs de personaliser la méthode de sélection des parents, du croisement et de mutation. Les parametres ainsi configurer offrent la posibilité d'expérimenter différentes configurations et méthodes adaptant l'algorithme aux besoins spécifiques de chaque utilisateur.
### Avec la surveillance de la stagnation l'algorithme favorise une exploration continue de l'espace de recherche offrant ainsi des chances supplementaires afin de trouver des solutions optimales.
###  Son architecture modulaire ainsi que ses option de configuration et visualisation des résutats en font un outil puissant pour résoudre des problèmes similaire.

