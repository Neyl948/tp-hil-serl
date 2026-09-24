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

**Q1.2** ...

**Q1.3** ...

**Q1.4** Success rate: ... · Mean time to success: ... s · Hardest phase: ...

---

## Part 2: Record demonstrations

Episodes recorded: ... · Successful: ... · Mean length: ... s

**Q2.1** ...

**Q2.2** ...

**Q2.3** ...

---

## Part 3: RL baseline without interventions

**Q3.1** ...

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
