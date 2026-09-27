# TP HIL-SERL: report

**Group:** 
**Students:** 
**Date:** 
**Device used (from `check_setup.py`):** cuda / mps / cpu — GPU model if any:

Replace every `...` with your answer. Insert figures from `runs/plots/` with `![caption](runs/plots/<file>.png)`. Keep the report under 6 pages when exported to PDF.

---

## Part 1: Discover the environment

**Human trials (1.2)**

| Operator | Attempt | Success (y/n) | Time (s) | What went wrong |
| --- | --- | --- | --- | --- |
| | 1 | | | |
| | 2 | | | |
| | 3 | | | |
| | 4 | | | |
| | 5 | | | |
| | 1 | | | |
| | 2 | | | |
| | 3 | | | |
| | 4 | | | |
| | 5 | | | |

**Q1.1** L'espace d'observation est composé des images de caméra 'front' et 'wrist' (tous les deux de dimension (128, 128, 3)) et de 'agent_pos', la position du robot (de dimension 18).  
Les 18 valeurs contiennent les informations sur la position des éléments du robot dans l'espace et sur leur vitesse.  
Pour l'action, l'espace brut du simulateur commande directement les 7 moteurs articulaires du bras et la pince. Avec le wrapper, l'espace vu par l'agent est réduit à 4 valeurs : les déplacements relatifs de la pince en $x$, $y$, $z$ ($\Delta x, \Delta y, \Delta z$) et une commande d'ouverture/fermeture du gripper.   La différence est que le wrapper masque toute la complexité articulaire : il calcule la cinématique inverse en interne, ce qui permet à l'agent de commander des mouvements 3D intuitifs plutôt que de devoir apprendre à coordonner chaque moteur du bras.

**Q1.2** Au lieux d'avoir 7 dimensions pour les articulations, l'espace de l'effecteur est réduit à 3 dimensions $\Delta x, \Delta y, \Delta z$. L'apprentissage est plus rapide car les actions sont moins complexes : si on veut faire un mouvement, on avance selon les 3 axes au lieu de calculer la rotation pour chaque articulation.

**Q1.3** La récompense n'est donnée que si le cube est attrapé puis soulevé. Elle est binaire : le robot a soulevé le cube ou non (1.0 ou 0.0).  
La récompense est sparse : le robot ne reçoit rien tant que la tâche n'est pas accomplie.  
Le problème en exploration aléatoire est que le robot n'apprend rien. Comme il n'est récompensé que si il a soulevé le cube, il faut qu'il y ait un enchaînement de mouvements qui lui permette d'effectuer cette tâche. Or en ne récoltant que des 0.0, la probabilité pour que ça arrive par hasard est faible.

**Q1.4** Success rate: 100% · Mean time to success: 2.4 s · Hardest phase: Aucune car le robot commence juste au dessus du cube, il faut juste descendre, attraper et remonter.

---

## Part 2: Record demonstrations

Episodes recorded: 10 · Successful: 6 · Mean length: 5.98 s

**Q2.1** Un algorithme off-policy apprend à partir de transitions stockées dans un buffer, peu importe qui a fait le geste. Il regarde juste ce qui s'est passé après chaque action, donc on peut lui donner des démonstrations faites par un humain. À l'inverse, un algorithme on-policy a besoin que les données viennent de sa propre politique au moment où il s'entraîne. Si on lui injecte des trajectoires humaines, ses calculs de mise à jour sont faussés parce que les données ne correspondent pas à ce que le modèle produit actuellement.

**Q2.2** 
* Propriétés utiles : une trajectoire directe et fluide vers le cube, qui offre un chemin court et sans bruit vers la récompense. Des corrections de trajectoire face à un désalignement, pour apprendre à la politique à récupérer en cas d'erreur.  
* Propriétés néfastes : des actions contradictoires ou hésitantes dans des états similaires, ce qui perturbe l'optimisation de la fonction $Q$. Des fermetures de pince dans le vide ou des temps morts prolongés, qui renforcent des actions inutiles ou pénalisées.

**Q2.3** 10 démos ne suffisent pas pour avoir une politique solide et robuste face au bruit. Cela expose le modèle au problème du covariate shift : la distribution des états rencontrés en test dévie de celle vue à l'entrainement. Comme le modèle n'a vu que des réussites parfaites, la moindre erreur le place dans un état inconnu. Ne sachant pas comment s'en rattraper, ses erreurs s'accumulent à chaque pas de temps et le bras finit par rater le cube.

---

## Part 3: RL baseline without interventions

**Q3.1** La température $alpha$ contrôle l'équilibre entre exploration et exploitation. Si $alpha$ est trop élevée, le robot cherche uniquement à maximiser son entropie : ses mouvements deviennent aléatoires et il n'apprend jamais à accomplir la tâche. Si $alpha$ est trop faible, l'entropie est négligée : la politique devient figée et arrête d'explorer.

**Q3.2** ...

**Q3.3** ...

**Q3.4** ...

**Q3.5** γ¹⁰⁰ = ... · Implication: ...

---

## Part 4: HIL-SERL with interventions

![noHIL vs HIL](runs/plots/GROUP_noHIL_vs_HIL.png)

| Run | First success (min) | Min to rolling reward ≥ 0.8 | Interventions | Human effort (s) |
| --- | --- | --- | --- | --- |
| noHIL | | | — | — |
| HIL | | | | |

**Q4.1** ...

**Q4.2** ...

**Q4.3** Metric proposed: ... · Value for our HIL run: ...

**Q4.4** ...

**Q4.5** ...

---

## Part 5: Experiment ___

**Q5.1 Hypothesis (written before the run):** ...

![HIL vs experiment](runs/plots/GROUP_HIL_vs_expX.png)

| Run | First success (min) | Min to rolling reward ≥ 0.8 | Interventions | Human effort (s) |
| --- | --- | --- | --- | --- |
| HIL (first 20 min) | | | | |
| exp ___ | | | | |

**Q5.1 Result:** ...

**Q5.2** ...

---

## Part 6: Class comparison

![Class results](runs/plots/class_results.png)

**Q6.1** ...

**Q6.2** ...

**Q6.3** ...

**Q6.4** ...

**Q6.5** ...
