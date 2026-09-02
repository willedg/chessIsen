# AlphaChess - Projet d'Échecs Deep Learning (M2 ISEN)

Ce projet implémente un moteur d'échecs basé sur l'architecture AlphaZero, combinant un réseau de neurones profond (SE-ResNet) et une recherche arborescente de Monte Carlo (MCTS) optimisée en C++.

## Fonctionnalités Clés
- **Architecture Nitro** : Réseau de neurones avec blocs Squeeze-and-Excitation pour une meilleure attention.
- **MCTS Hybride** : Cœur de recherche écrit en C++ (PyBind11) pour des performances de calcul optimales (NPS élevé).
- **Pipeline de Self-Play** : Génération autonome de données d'entraînement et boucle d'apprentissage automatisée.
- **Interface GUI** : Outil de test interactif avec positions de mat prédéfinies (M1 à M3).

## Structure du Projet
```text
Projet échec/
├── progress.json                  # Suivi de l'état de l'entraînement (cycle/phase)
├── setup.py                       # Configuration de la compilation C++ (PyBind11)
├── train_log.csv                  # Logs d'entraînement (pertes Total/Policy/Value)
├── eval_log.csv                   # Logs d'évaluation (winrate vs modèle précédent)
├── training_progress.png          # Visualisation graphique de la progression
├── test_proven_dtm.py             # Benchmarks de performance et profondeur
│
├── wp1_env.py                     # Environnement de base et règles
├── wp2/                           # ARCHITECTURE DU MODÈLE (AlphaChessNet)
├── wp3/                           # ENGINE & OPTIMISATION C++ (MCTS, Predictor)
├── wp4/                           # TRAINING PIPELINE (Trainer, Self-play)
└── wp5/                           # CYCLE MANAGEMENT & UI (Orchestrateur, GUI)
```

## Installation et Utilisation

### 1. Compilation du module C++
```powershell
python setup.py build_ext --inplace
```

### 2. Lancer l'entraînement (Boucle Automatique)
```powershell
python -m wp5.cycle --cycles 20
```

### 3. Jouer ou tester le modèle (GUI)
```powershell
python -m wp5.play_gui --model checkpoints/current.pt --device cuda --n_sim 1600
```

## Utilisation de l'Interface Graphique (GUI)

L'interface permet de jouer contre le modèle ou de tester ses capacités sur des positions spécifiques.

### Modes de Jeu
Au lancement, une fenêtre propose plusieurs options :
- **Human vs Model** : Pour affronter l'IA directement. Vous choisissez votre couleur au démarrage.
- **Model vs Random** : Le modèle affronte un joueur qui joue des coups aléatoires.
- **Model vs Stockfish** : Le modèle affronte le moteur Stockfish (niveau 0 par défaut pour les tests).
- **Load Position (FEN)** : Permet de charger une position spécifique (ex: Mat en 1, 2 ou 3) pour tester la résolution du modèle.

### Fonctionnalités de Test et Debug
Pendant le jeu, des contrôles supplémentaires sont disponibles :
- **Model Move** : Force le modèle à calculer et jouer le meilleur coup pour la position actuelle. Utile pour voir ce que l'IA propose sans forcément lancer une partie complète.
- **Stockfish Move** : Demande à Stockfish de jouer le meilleur coup théorique. Permet de tester la défense du modèle ou de simuler des séquences précises.
- **Simulations Slider** : Ajuste en temps réel le nombre de simulations MCTS. Plus le nombre est élevé (ex: 3200), plus le modèle réfléchit profondément mais lentement.
- **Reset / New Game** : Réinitialise l'échiquier et les moteurs.
- **Quit** : Ferme proprement l'application et libère les processus en arrière-plan (Stockfish, Cuda).

### Affichage Technique
- **NPS (Nodes Per Second)** : Indique la vitesse de calcul de la recherche MCTS sur votre matériel.
- **Info Label** : Affiche les informations sur les joueurs et le trait actuel.

## Analyse des Résultats
Le projet inclut des outils pour générer des rapports de progression. Un historique de 29 cycles d'entraînement est disponible, montrant la convergence de la perte et l'évolution du taux de victoire contre les versions précédentes du modèle.

---
*Projet réalisé dans le cadre du Master 2 - ISEN (2025-2026)*
