# Scoring_Linux
Projet réalisé avec [Yasmine Ouyahya](https://github.com/youyahya) et [Amira Slimene](https://github.com/aslimene).

Le but de ce projet est de réaliser une grille de score permettant d'aider à la décision.

Ainsi, on a créé une grille de score de score est calibrée sur 1000 points, plus le score est élevé, moins le client a de risque de faire défaut. On fixe les Odds à $\frac{1}{500}$ (500:1) et les PDO (Points Double the Odds) à 30 points. 
Une personne qui a un  score 1000 points  a une probabilité de défaut de 1/500 alors qu'une personne qui a un score de 1030 (1000 + 30) elle a une une probabilité de défaut égale à $\frac{1}{1000}$ (1000:1 = 2*500:1).
