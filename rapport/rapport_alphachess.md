# AlphaChess — Moteur d'Échecs par Apprentissage par Renforcement

**Projet de fin d'études — ISEN M2**

**Auteur :** [Votre Nom]
**Date :** Février 2026
**Encadrant :** [Nom de l'encadrant]

---

## Table des matières

- Résumé
- Introduction générale
- I. Mise en place de l'environnement de jeu (WP1)
- II. Conception du réseau de neurones (WP2)
- III. Implémentation du Monte Carlo Tree Search (WP3)
- IV. Auto-apprentissage par self-play et entraînement (WP4)
- V. Évaluation, progression et itérations (WP5)
- VI. Difficultés, limites et pistes d'amélioration
- VII. Gestion du projet et chronologie
- Conclusion générale
- Bibliographie
- Annexes

---

## Résumé

Le projet AlphaChess vise à concevoir un moteur d'échecs autonome capable d'apprendre à jouer sans connaissance humaine préalable, en s'appuyant sur les principes de l'apprentissage par renforcement profond. Entièrement entraîné sur une carte graphique grand public **NVIDIA RTX 3070ti**, le système combine un réseau de neurones convolutif résiduel à blocs Squeeze-and-Excitation (SE-ResNet) et une recherche arborescente de Monte-Carlo (MCTS) implémentée en C++ pour des performances optimales. Le pipeline d'auto-apprentissage génère ses propres données d'entraînement via un processus de « self-play », dans lequel le modèle joue contre lui-même pour découvrir de nouvelles stratégies. Après 29 cycles complets d'entraînement, le modèle démontre une capacité à résoudre des problèmes tactiques simples (mat en 1, mat en 2) et une amélioration mesurable de sa compréhension stratégique. Ce rapport détaille l'intégralité de la chaîne technique, de l'encodage du plateau jusqu'à la boucle de progression itérative, et explore les pistes d'amélioration telles que l'injection de parties de Grands Maîtres ou l'entraînement ciblé sur des puzzles tactiques.

**Mots-clés :** Apprentissage par renforcement, Réseaux de neurones profonds, Monte Carlo Tree Search, Échecs, AlphaZero, Self-Play, PyTorch, C++.

---

## Introduction générale

### Contexte et motivation

Depuis la victoire historique de Deep Blue contre Garry Kasparov en 1997, l'intelligence artificielle appliquée aux échecs a connu une transformation radicale. Pendant deux décennies, les moteurs d'échecs comme Stockfish ont dominé grâce à des algorithmes de recherche alpha-bêta combinés à des fonctions d'évaluation heuristiques minutieusement ajustées par des experts humains. Ces fonctions attribuaient des valeurs numériques à des concepts stratégiques tels que la structure de pions, la sécurité du roi ou le contrôle du centre, résultant en des moteurs extrêmement puissants mais fondamentalement dépendants de la connaissance humaine codée en dur.

En décembre 2017, DeepMind a bouleversé ce paradigme avec la publication d'AlphaZero [1]. Ce système a démontré qu'un réseau de neurones profond, combiné à une recherche arborescente de Monte-Carlo (MCTS) et entraîné exclusivement par auto-apprentissage (self-play), pouvait non seulement égaler mais surpasser les meilleurs moteurs traditionnels, et ce en seulement quelques heures d'entraînement sur du matériel spécialisé (TPU). AlphaZero ne possédait aucune connaissance des échecs au-delà des règles du jeu : pas de tables d'ouverture, pas de fonctions d'évaluation heuristiques, pas de bases de données de parties de Grands Maîtres. L'ensemble de sa compréhension stratégique et tactique émergeait uniquement du processus d'auto-apprentissage.

Ce résultat a ouvert la voie à une nouvelle génération de moteurs d'échecs neuronaux, dont le plus notable est Leela Chess Zero (Lc0), un projet open-source communautaire qui reproduit l'approche d'AlphaZero sur du matériel grand public grâce à un effort distribué.

Le présent projet, baptisé **AlphaChess**, s'inscrit dans cette lignée. L'objectif est de comprendre, implémenter et valider l'ensemble de la chaîne algorithmique d'AlphaZero, depuis la représentation du plateau jusqu'à la boucle d'auto-amélioration, en utilisant exclusivement du matériel accessible (une NVIDIA RTX 3070ti).

### Objectifs du projet

Les objectifs du projet sont les suivants :

1. **Développer un environnement de jeu complet** capable de modéliser les règles des échecs, de générer les coups légaux et de détecter les conditions terminales.
2. **Concevoir et implémenter un réseau de neurones** à double tête (Policy et Value) capable d'évaluer une position et de suggérer des coups.
3. **Implémenter un moteur MCTS performant** en C++ pour guider la recherche arborescente en utilisant les prédictions du réseau.
4. **Mettre en place un pipeline d'auto-apprentissage** complet, incluant la génération de données par self-play, l'entraînement du réseau et l'évaluation comparative des modèles.
5. **Analyser les performances** du système et identifier les leviers d'amélioration.

### Problématique

Comment concevoir et optimiser un moteur d'échecs neuronal sur du matériel grand public (RTX 3070ti), capable d'apprendre par lui-même via le self-play, tout en maintenant un rythme de progression mesurable malgré les contraintes de puissance de calcul ?

### Organisation du rapport

Le rapport est structuré en sept chapitres correspondant aux différents Work Packages (WP) du projet :

- Le **Chapitre I** décrit la modélisation de l'environnement de jeu.
- Le **Chapitre II** présente l'architecture du réseau de neurones.
- Le **Chapitre III** détaille l'implémentation du MCTS.
- Le **Chapitre IV** explique le mécanisme de self-play et d'entraînement.
- Le **Chapitre V** analyse les résultats obtenus sur 29 cycles.
- Le **Chapitre VI** discute des difficultés rencontrées et des pistes d'amélioration.
- Le **Chapitre VII** présente la gestion du projet et sa chronologie.

---

## I. Mise en place de l'environnement de jeu (WP1)

### 1. Modélisation du jeu d'échecs

Le jeu d'échecs se caractérise par sa complexité combinatoire : on estime le nombre de parties possibles à environ 10^120 (nombre de Shannon). Cette immensité rend impossible toute approche par force brute et nécessite des méthodes d'approximation intelligentes.

L'environnement de jeu repose sur la bibliothèque **python-chess** (version 1.x), qui fournit :

- Une représentation bitboard du plateau, optimisée pour les opérations binaires rapides.
- Un générateur de coups légaux conforme aux règles officielles de la FIDE.
- La gestion complète des cas spéciaux : roque, prise en passant, promotion, pat, répétition triple, règle des 50 coups.
- L'import/export au format PGN (Portable Game Notation) et FEN (Forsyth-Edwards Notation).

Le choix de python-chess, plutôt qu'une implémentation manuelle, permet de se concentrer sur les aspects d'apprentissage automatique tout en garantissant la conformité réglementaire de l'environnement.

#### Exemple de représentation FEN

La position initiale est représentée par la chaîne FEN suivante :

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

Cette notation encode, de gauche à droite : la disposition des pièces (rang par rang), le trait (w = blancs), les droits au roque (KQkq), la case de prise en passant (-), le compteur de demi-coups pour la règle des 50 coups (0) et le numéro du coup (1).

### 2. Représentation de l'état du jeu

Pour qu'un réseau de neurones puisse « comprendre » une position d'échecs, il est nécessaire de la convertir en un tenseur numérique. Notre encodeur transforme un objet `chess.Board` en un tenseur de dimensions **(19, 8, 8)**, où chaque plan 8×8 correspond à une caractéristique du plateau :

| Plans | Description |
|-------|-------------|
| 0-5   | Pions, Cavaliers, Fous, Tours, Dames, Rois **blancs** |
| 6-11  | Pions, Cavaliers, Fous, Tours, Dames, Rois **noirs** |
| 12    | Trait (1.0 si blancs, 0.0 si noirs) |
| 13    | Droit au petit roque blanc |
| 14    | Droit au grand roque blanc |
| 15    | Droit au petit roque noir |
| 16    | Droit au grand roque noir |
| 17    | Compteur de demi-coups (normalisé /100) |
| 18    | Indicateur de répétition |

**Table 1** : Encodage du plateau en 19 plans (8×8).

Chaque plan de pièce est un masque binaire : la case (rank, file) vaut 1.0 si la pièce correspondante s'y trouve, 0.0 sinon. Les plans de droits au roque et de trait sont des plans constants (toutes les cases à la même valeur).

#### Implémentation de l'encodeur

L'encodeur existe en deux versions :

1. **Version Python** (`wp2/encoders.py`) : Utilisée pour le prototypage et la validation.
2. **Version C++** (`wp3/board_encoder.hpp`) : Version haute performance intégrée au module MCTS, utilisant les bitboards natifs de python-chess pour un encodage ultra-rapide.

```python
# Extrait de wp2/encoders.py
def board_to_tensor(board: chess.Board):
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank = 7 - (square // 8)
        file = square % 8
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + PIECE_PLANES[piece.piece_type], rank, file] = 1.0
    # ... plans additionnels (trait, roque, etc.)
    return torch.tensor(planes, dtype=torch.float32)
```

La version C++ (`get_fast_state_info`) récupère toutes les informations en un seul appel Python, puis effectue l'encodage côté C++ via des opérations bitwise, offrant un gain de performance de l'ordre de 5x à 10x par rapport à la version Python.

### 3. Définition des actions et conditions terminales

#### Espace d'actions

L'espace d'actions est un vecteur de taille fixe **4672**. Ce choix est dérivé de la méthode utilisée par AlphaZero :

- **Indices 0 à 4095** : Coups normaux encodés comme `from_square × 64 + to_square`. Cela couvre toutes les combinaisons de cases de départ et d'arrivée.
- **Indices 4096 à 4671** : Coups de promotion, encodés avec la pièce de promotion, la colonne de départ, la colonne d'arrivée et la couleur.

```python
# Extrait de wp2/action_map.py
ACTION_SPACE = 4672

def move_to_index(move: chess.Move) -> int:
    if move.promotion:
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1,
                     chess.ROOK: 2, chess.QUEEN: 3}
        offset = (1 if is_black else 0) * 256 + p_idx * 64 + from_file * 8 + to_file
        return 4096 + offset
    return move.from_square * 64 + move.to_square
```

La fonction inverse `index_to_move` permet de reconstruire un objet `chess.Move` à partir d'un indice, ce qui est nécessaire pour appliquer le coup choisi par le MCTS.

#### Conditions terminales

Le résultat d'une partie est encodé par un scalaire :
- **z = +1** : Victoire des blancs
- **z = 0** : Partie nulle (pat, répétition, règle des 50 coups, matériel insuffisant)
- **z = -1** : Victoire des noirs

Ce scalaire est utilisé rétrospectivement pour étiqueter chaque position de la partie avec le résultat final, créant ainsi le signal d'apprentissage pour la tête Value du réseau.

---

## II. Conception du réseau de neurones (WP2)

### 1. Principes du Reinforcement Learning

L'apprentissage par renforcement (RL) est un paradigme d'apprentissage automatique dans lequel un agent apprend à prendre des décisions optimales en interagissant avec un environnement. Contrairement à l'apprentissage supervisé classique, l'agent ne dispose pas d'exemples étiquetés « coup optimal » : il doit découvrir les meilleures stratégies par essai-erreur.

Dans le contexte d'AlphaChess, l'agent est le réseau de neurones, l'environnement est le plateau d'échecs, et le signal de récompense est le résultat final de la partie. L'originalité de l'approche AlphaZero réside dans le fait que l'agent génère ses propres données d'entraînement en jouant contre lui-même (self-play), puis améliore sa politique et son évaluation en apprenant de ces parties.

Le processus suit le schéma suivant :

1. Le réseau produit une estimation initiale (aléatoire au départ).
2. Le MCTS affine cette estimation en simulant des milliers de variantes.
3. La distribution de visites du MCTS constitue une « politique améliorée » ($\pi$).
4. Le réseau est entraîné à reproduire cette politique améliorée.
5. Le cycle recommence avec un réseau plus précis.

Ce mécanisme de « distillation de politique » est la clé de la convergence : le MCTS agit comme un « professeur » qui guide le réseau vers des décisions de plus en plus pertinentes.

### 2. Architecture du réseau (Policy / Value)

Le réseau **AlphaChessNet** suit une architecture à tronc commun et double tête, inspirée d'AlphaZero. Le tronc partagé capture les caractéristiques générales de la position, tandis que les deux têtes se spécialisent dans des tâches distinctes.

#### Tronc commun (Backbone)

Le tronc est composé de :

1. **Couche d'entrée** : Une convolution 3×3 transformant les 19 plans d'entrée en 128 canaux, suivie d'une BatchNorm et d'une activation ReLU.
2. **Tour résiduelle** : 12 blocs résiduels SE-ResNet empilés séquentiellement.

#### Bloc Résiduel SE (Squeeze-and-Excitation)

Chaque bloc résiduel combine :
- Deux convolutions 3×3 avec BatchNorm et ReLU.
- Une connexion résiduelle (skip connection) pour éviter le problème de disparition du gradient.
- Un module **Squeeze-and-Excitation (SE)** qui recalibre dynamiquement l'importance de chaque canal.

Le module SE fonctionne en trois étapes :
1. **Squeeze** : Moyenne globale (Global Average Pooling) réduisant chaque canal à un scalaire.
2. **Excitation** : Deux couches linéaires avec réduction (facteur 16) apprenant les interdépendances entre canaux.
3. **Recalibration** : Multiplication du tenseur original par les poids appris.

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

#### Tête Policy (π)

La tête Policy produit un vecteur de probabilités sur les 4672 actions possibles :
1. Convolution 1×1 réduisant à 32 canaux.
2. BatchNorm + ReLU.
3. Aplatissement (flatten) vers un vecteur de taille 32×8×8 = 2048.
4. Couche linéaire finale vers 4672 logits.

Les logits sont ensuite filtrés par le masque de coups légaux et normalisés par softmax dans le MCTS.

#### Tête Value (v)

La tête Value produit un scalaire dans [-1, 1] estimant la probabilité de victoire :
1. Convolution 1×1 réduisant à 32 canaux.
2. BatchNorm + ReLU.
3. Aplatissement vers 2048.
4. Couche linéaire vers 256 neurones + ReLU.
5. Couche linéaire finale vers 1 neurone + tanh.

La fonction tanh borne naturellement la sortie dans l'intervalle [-1, 1], ce qui correspond à notre encodage des résultats.

### 3. Choix techniques et justifications

| Choix technique | Justification |
|----------------|---------------|
| **SE-ResNet** au lieu de ResNet simple | Les blocs SE permettent au réseau de pondérer dynamiquement l'importance des différents canaux de caractéristiques, améliorant la représentation avec un coût minimal (+2% de paramètres). |
| **12 blocs résiduels** | Compromis entre profondeur (capacité représentationnelle) et vitesse d'inférence. AlphaZero utilise 19 ou 39 blocs, mais notre matériel impose une architecture plus légère. |
| **128 canaux** | Taille suffisante pour capturer les patterns échiquéens sans saturer la mémoire GPU. |
| **BatchNorm** | Stabilise l'entraînement des réseaux profonds en normalisant les activations. |
| **Mixed Precision (FP16)** | Divise par 2 la consommation mémoire et accélère les calculs sur les GPU NVIDIA (architecture Ampere). |
| **TorchScript Tracing** | Compilation du graphe d'exécution pour éliminer l'overhead Python lors de l'inférence, offrant un gain de 20-30% de vitesse. |

**Table 2** : Justification des choix techniques pour le réseau.

Le nombre total de paramètres du réseau est d'environ **13,5 millions**, ce qui le rend suffisamment expressif pour capturer les subtilités positionnelles tout en permettant une inférence rapide sur la RTX 3070ti.

---

## III. Implémentation du Monte Carlo Tree Search (WP3)

### 1. Principe général du MCTS

Le Monte Carlo Tree Search est un algorithme de recherche arborescente qui utilise des simulations aléatoires pour évaluer les positions de jeu. Contrairement à l'algorithme minimax classique (utilisé par Stockfish), le MCTS ne nécessite pas de fonction d'évaluation heuristique : il s'appuie sur les prédictions du réseau de neurones pour guider sa recherche.

Le MCTS fonctionne en quatre phases répétées :

1. **Sélection** : En partant de la racine, on descend dans l'arbre en choisissant à chaque nœud l'enfant qui maximise la formule PUCT (Predictor Upper Confidence bound applied to Trees).
2. **Expansion** : Lorsqu'on atteint un nœud non encore développé (feuille), on l'évalue avec le réseau de neurones pour obtenir la politique (π) et la valeur (v).
3. **Backpropagation** : La valeur obtenue est propagée vers la racine en mettant à jour les statistiques de chaque nœud traversé.
4. **Choix du coup** : Après un nombre fixe de simulations, le coup ayant reçu le plus de visites est sélectionné.

#### Formule PUCT

La sélection utilise la formule suivante pour choisir l'enfant optimal :

$$a^* = \arg\max_a \left[ Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} \right]$$

Où :
- $Q(s,a)$ : Valeur moyenne de l'action $a$ dans l'état $s$
- $P(s,a)$ : Prior (probabilité initiale) donné par le réseau
- $N(s)$ : Nombre total de visites du nœud parent
- $N(s,a)$ : Nombre de visites du nœud enfant
- $c_{puct}$ : Constante d'exploration (1.25 en self-play, 1.0 en évaluation)

Cette formule assure un équilibre dynamique entre exploitation (favoriser les coups ayant une bonne valeur moyenne Q) et exploration (essayer les coups peu visités mais ayant un prior élevé P).

### 2. Structure de l'arbre et rôle des nœuds

#### Architecture mémoire : l'Arena Allocator

Pour maximiser les performances, l'arbre MCTS est implémenté en C++ avec un système d'**Arena Allocation**. Au lieu d'allouer et désallouer des nœuds individuellement (ce qui est coûteux en temps), un grand bloc de mémoire contiguë est pré-alloué :

```cpp
class NodeArena {
    std::vector<Node> pool;
    uint32_t next_free = 0;
public:
    NodeArena(size_t initial_cap = 1000000) {
        pool.resize(initial_cap);
    }
    uint32_t allocate(float prior, int parent_idx, int move_idx) {
        if (next_free >= pool.size())
            pool.resize(pool.size() * 1.5);
        uint32_t idx = next_free++;
        pool[idx].reset(prior, parent_idx, move_idx);
        return idx;
    }
};
```

Chaque nœud stocke :
- **visit_count** : Nombre de simulations passant par ce nœud.
- **value_sum** : Somme des valeurs retro-propagées.
- **prior** : Probabilité initiale donnée par le réseau.
- **children** : Table de hachage des nœuds enfants (indexés par l'action).
- **is_terminal** : Indicateur de position terminale (mat, pat).
- **virtual_loss** : Compteur de « pertes virtuelles » pour la parallélisation.

#### Batch Processing

L'une des optimisations majeures est le traitement par lots (batching) des évaluations réseau. Au lieu d'évaluer une position à la fois, le MCTS collecte plusieurs feuilles et les envoie au réseau en un seul appel. Cela exploite pleinement le parallélisme du GPU :

```cpp
// Collecte de 64 feuilles en une seule passe
while (sims_done < n_sim) {
    int cur_batch = std::min(batch_size, n_sim - sims_done);
    // ... sélection de cur_batch feuilles
    py::list batch_boards;
    for (int idx : leaf_indices) batch_boards.append(board_pool[idx]);
    py::list outputs = predict(batch_boards); // Un seul appel GPU
    // ... backpropagation
}
```

### 3. Intégration du réseau de neurones au MCTS

#### Le Predictor

La classe `Predictor` (`wp3/predictor.py`) sert d'interface entre le MCTS C++ et le réseau PyTorch. Elle gère :

1. **Le chargement du modèle** : Support des fichiers `.pt` (PyTorch) et `.ts` (TorchScript).
2. **L'optimisation d'inférence** : Conversion automatique en FP16 et tracing TorchScript.
3. **Le cache de coups** : Pré-calcul et mise en mémoire de tous les objets Move et chaînes UCI pour éviter les allocations répétées.

```python
class Predictor:
    def __init__(self, model_or_path, device="cuda"):
        self.model = AlphaChessNet()
        checkpoint = torch.load(model_or_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        # Optimisations
        self.model = self.model.half()  # FP16
        self.model = torch.jit.trace(self.model, example_input)  # TorchScript
```

#### Bruit de Dirichlet

Pour encourager l'exploration pendant le self-play, du bruit de Dirichlet est ajouté aux priors de la racine :

$$P'(s,a) = (1 - \epsilon) \cdot P(s,a) + \epsilon \cdot \eta_a$$

Où $\eta \sim Dir(\alpha)$ avec $\alpha = 0.03$ et $\epsilon = 0.25$ pour le self-play.

Ce bruit est crucial pour la diversité des données d'entraînement : sans lui, le modèle jouerait toujours les mêmes parties, limitant considérablement son apprentissage.

#### Modes de fonctionnement

Le MCTS adapte son comportement selon le contexte d'utilisation :

| Paramètre | Self-Play | Évaluation | Jeu (GUI) |
|-----------|-----------|------------|-----------|
| Bruit Dirichlet (ε) | 0.25 | 0.03 | 0.0 |
| Température | 1.0 (20 premiers coups) | 0.0 | 0.0 |
| c_puct | 1.25 | 1.0 | 1.0 |
| Objectif | Explorer et diversifier | Déterminer le plus fort | Jouer au mieux |

**Table 3** : Paramètres MCTS selon le mode d'utilisation.

#### Early Exit

Une optimisation supplémentaire permet d'arrêter la recherche prématurément si un coup domine clairement :

```cpp
if (sims_done >= 400 && max_visits > second_max_visits * 3) {
    break; // Le meilleur coup est déjà déterminé
}
```

Cette heuristique économise du temps de calcul sans sacrifier la qualité des décisions.

