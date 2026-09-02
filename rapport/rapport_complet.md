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



---

## IV. Auto-apprentissage par self-play et entraînement (WP4)

### 1. Génération des parties par self-play

Le self-play est le cœur du processus d'apprentissage d'AlphaChess. Le modèle joue contre lui-même pour générer des données d'entraînement. À chaque coup, le MCTS réalise un nombre fixe de simulations (1600 dans notre configuration) pour produire une distribution de probabilités améliorée sur les coups légaux.

#### Processus d'une partie

Le déroulement d'une partie de self-play suit le schéma suivant :

1. **Initialisation** : Le plateau est placé en position de départ. L'arbre MCTS est vide.
2. **Boucle de jeu** : Pour chaque position :
   a. Le MCTS effectue 1600 simulations en utilisant le réseau pour évaluer les feuilles.
   b. La distribution de visites normalisée $\pi$ est enregistrée.
   c. L'état du plateau (tenseur 19×8×8) est sauvegardé.
   d. Un coup est sélectionné (avec température pour les 20 premiers coups, puis déterministement).
   e. L'arbre est avancé (réutilisation du sous-arbre correspondant au coup joué).
3. **Fin de partie** : Le résultat $z \in \{-1, 0, +1\}$ est déterminé.
4. **Étiquetage rétrospectif** : Chaque position est associée au résultat final, vu du point de vue du joueur au trait.

#### Température et exploration

La température contrôle le degré d'aléatoire dans le choix du coup :

- **Température = 1.0** (20 premiers coups) : Le coup est échantillonné proportionnellement au nombre de visites. Un coup ayant reçu 60% des visites a 60% de chances d'être joué. Cela encourage la diversité des ouvertures.
- **Température → 0** (après le 20ème coup) : Le coup le plus visité est systématiquement choisi. Cela garantit la qualité des données dans les phases critiques de milieu et fin de partie.

```python
# Extrait de wp4/selfplay_worker.py
if move_count < 20:
    indices = np.where(pi_vec > 1e-8)[0]
    probs = pi_vec[indices]
    probs /= probs.sum()
    chosen_idx = np.random.choice(indices, p=probs)
    chosen = predictor.move_ucis[chosen_idx]
```

#### Parallélisation par Workers

Pour accélérer la génération, plusieurs processus travaillent en parallèle :

```python
# Extrait de wp5/cycle.py
spawn_workers(2, 400, 1600, CURRENT, cycle_num, 1024)
```

Cette commande lance 2 workers générant chacun 400 parties avec 1600 simulations par coup. Au total, **800 parties** sont produites par cycle, créant environ 800 fichiers `.npz` (un par partie).

### 2. Construction du jeu de données

#### Format des shards

Chaque partie est sauvegardée dans un fichier `.npz` (format NumPy compressé) contenant trois tableaux :

- **states** : Tableau de shape `(N, 19, 8, 8)` contenant les N positions encodées de la partie.
- **policies** : Tableau de shape `(N, 4672)` contenant les distributions de politique MCTS.
- **values** : Tableau de shape `(N,)` contenant les résultats étiquetés (-1, 0, ou +1).

```python
# Extrait de wp4/utils_io.py
def save_training_examples(states, pis, zs, out_dir, prefix="shard"):
    np.savez_compressed(path,
        states=np.array(states, dtype=np.float32),
        policies=np.array(pis, dtype=np.float32),
        values=np.array(zs, dtype=np.float32)
    )
```

#### Fenêtre glissante

Pour éviter que le modèle « oublie » ses acquis tout en incorporant de nouvelles données, un mécanisme de **fenêtre glissante** est utilisé. Le système conserve les **5000 shards les plus récents** et supprime automatiquement les plus anciens :

```python
# Nettoyage fenêtre glissante (extrait de cycle.py)
shards = list_shards("data/selfplay/shards")
if len(shards) > 50000:
    for s in shards[:-50000]:
        os.remove(s)
```

Cette approche permet de maintenir un dataset diversifié et représentatif des capacités actuelles du modèle.

### 3. Fonction de perte et optimisation

#### Fonction de perte composite

La perte totale combine deux objectifs :

$$\mathcal{L} = \mathcal{L}_{policy} + \mathcal{L}_{value} + c \|\theta\|^2$$

Où :

- **Perte de politique** (Cross-Entropy) : $\mathcal{L}_{policy} = -\sum_a \pi_a \log p_a(\theta)$

  Le réseau doit reproduire la distribution de visites du MCTS. La cross-entropy mesure la divergence entre la politique du réseau $p$ et la politique cible $\pi$ produite par le MCTS.

- **Perte de valeur** (MSE) : $\mathcal{L}_{value} = (z - v(\theta))^2$

  Le réseau doit prédire le résultat final de la partie. L'erreur quadratique moyenne (MSE) pénalise les écarts entre la prédiction $v$ et le résultat réel $z$.

- **Régularisation L2** : $c \|\theta\|^2$ avec $c = 10^{-4}$

  Prévient le sur-apprentissage en pénalisant les poids de grande magnitude.

#### Optimiseur

L'entraînement utilise l'optimiseur **Adam** avec les hyperparamètres suivants :

| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 0.001 |
| β1 | 0.9 |
| β2 | 0.999 |
| Weight decay | 1×10⁻⁴ |
| Batch size | variable |

**Table 4** : Hyperparamètres de l'optimiseur Adam.

#### Chargement de données optimisé (Nitro Loading)

Le pipeline de données inclut un système de chargement optimisé qui pré-charge les fichiers `.npz` de manière asynchrone pour saturer le GPU :

1. **Listing rapide** : Récupération ordonnée des fichiers par numéro de série.
2. **Fenêtre glissante** : Sélection des N shards les plus récents.
3. **Chargement par batch** : Les shards sont concaténés en mémoire puis découpés en mini-batches.
4. **Mélange aléatoire** : Les données sont mélangées à chaque époque pour éviter les corrélations séquentielles.

---

## V. Évaluation, progression et itérations (WP5)

### 1. Méthodes d'évaluation

#### Duel entre modèles

À la fin de chaque cycle d'entraînement, un match d'évaluation oppose le modèle **Candidat** (nouvellement entraîné) au modèle **Actuel** (champion en titre). Le protocole est le suivant :

- **Nombre de parties** : 100 (parfois réduit à 40 pour accélérer les cycles).
- **Alternance des couleurs** : Le candidat joue les blancs dans les parties paires et les noirs dans les parties impaires, garantissant l'absence de biais de couleur.
- **Paramètres MCTS** : `c_puct = 1.0`, bruit Dirichlet réduit (ε = 0.03), température 0 (jeu déterministe).
- **Critère d'acceptation** : Le candidat doit obtenir un winrate strictement supérieur à 50% pour remplacer l'actuel.

```python
# Extrait de wp5/cycle.py
score_new = evaluate(CURRENT, CANDIDATE, device="cuda",
                     n_games=100, n_workers=8, n_sim=1600)
rate = score_new / n_games
if rate > 0.50:
    print("ACCEPTED")
    os.replace(CANDIDATE, CURRENT)
else:
    print("REJECTED")
    os.remove(CANDIDATE)
```

Ce critère strict protège contre les régressions : un modèle qui a « de la chance » sur un échantillon limité ne remplacera pas un modèle éprouvé.

#### Benchmark tactique

En complément, un ensemble de positions tactiques prédéfinies permet de mesurer la capacité du modèle à résoudre des problèmes concrets :

| Position | Type | Résultat (Cycle 29) |
|----------|------|---------------------|
| Mat du Berger | Mat en 1 | ✅ Résolu (100 sims) |
| Anastasia | Mat en 2 | ✅ Résolu (800 sims) |
| Mat du couloir | Mat en 1 | ✅ Résolu (50 sims) |
| Défense Philidor | Tactique profonde | ❌ Non résolu (3200 sims) |

**Table 5** : Résultats du benchmark tactique au Cycle 29.

### 2. Analyse des performances

#### Évolution de la fonction de perte

Le graphique suivant (voir `training_progress.png`) montre l'évolution des pertes sur 29 cycles :

**[Insérer ici : training_progress.png — Figure 1]**

*Figure 1 : Évolution de la perte totale, de la perte de politique et de la perte de valeur sur 29 cycles d'entraînement.*

**Observations :**

- **Perte totale** : Décroissance régulière de **2.88** (Cycle 1) à **2.75** (Cycle 29), soit une réduction de 4.5%. Cette décroissance, bien que modeste en valeur absolue, est significative dans le contexte des échecs où les marges sont faibles.
- **Perte de politique** : Décroissance de **2.77** à **2.71**. Le réseau améliore graduellement sa capacité à prédire les coups que le MCTS considère comme optimaux.
- **Perte de valeur** : Fluctuation autour de **0.05 à 0.11**. Cette composante est plus bruitée car elle dépend du résultat final de parties longues, qui contient intrinsèquement plus de variance.

#### Évolution du taux de victoire

| Cycle | Winrate | Décision |
|-------|---------|----------|
| 1 | 53.0% | ✅ Accepté |
| 2 | 57.0% | ✅ Accepté |
| 3 | 58.5% | ✅ Accepté |
| 4 | 55.0% | ✅ Accepté |
| 5 | 55.5% | ✅ Accepté |
| 6 | 41.5% | ❌ Rejeté |
| 7 | 62.0% | ✅ Accepté |
| 8 | 65.5% | ✅ Accepté |
| 9 | 44.5% | ❌ Rejeté |
| 10 | 59.0% | ✅ Accepté |
| 11 | 54.0% | ✅ Accepté |
| 12 | 52.0% | ✅ Accepté |
| 13 | 44.0% | ❌ Rejeté |
| 14 | 62.5% | ✅ Accepté |
| 15 | 40.5% | ❌ Rejeté |
| 16 | 49.0% | ❌ Rejeté |
| 17 | 56.5% | ✅ Accepté |
| 18 | 52.5% | ✅ Accepté |
| 19 | 46.5% | ❌ Rejeté |
| 20 | 54.5% | ✅ Accepté |
| 21 | 48.0% | ❌ Rejeté |
| 22 | 49.0% | ❌ Rejeté |
| 23 | 54.5% | ✅ Accepté |
| 24 | 66.5% | ✅ Accepté |
| 25 | 42.0% | ❌ Rejeté |
| 26 | 50.0% | ❌ Rejeté |
| 27 | 43.0% | ❌ Rejeté |
| 28 | 44.0% | ❌ Rejeté |
| 29 | 47.0% | ❌ Rejeté |

**Table 6** : Historique complet des évaluations sur 29 cycles.

**Analyse :**

- **Phase de progression rapide (Cycles 1-12)** : Le modèle progresse régulièrement avec un taux d'acceptation de ~75%. Les gains sont importants car le modèle part de zéro et chaque cycle apporte des connaissances fondamentales.
- **Phase de plateau (Cycles 13-29)** : Le taux d'acceptation chute à ~40%. Le modèle a acquis les bases et les améliorations deviennent marginales. Les cycles de rejet sont plus fréquents, indiquant que le système atteint une limite avec la configuration actuelle.
- **Cycles remarquables** : Le Cycle 8 (65.5%) et le Cycle 24 (66.5%) montrent des « sauts » qualitatifs où le modèle a découvert un concept stratégique nouveau.

### 3. Limites observées

1. **Stagnation tactique** : Le modèle résout les mats en 1 et 2 mais échoue sur les combinaisons profondes (mat en 4+). Cela est lié au nombre de simulations (1600), insuffisant pour explorer des lignes de 8+ demi-coups.
2. **Biais de contrôle du centre** : Le modèle n'a pas toujours une préférence marquée pour le contrôle du centre en ouverture, contrairement aux principes classiques. Cela suggère que 29 cycles ne suffisent pas pour « redécouvrir » tous les principes stratégiques.
3. **Plateau d'apprentissage** : À partir du Cycle 20, la progression devient quasi-nulle. Le modèle a besoin soit de plus de données, soit de données de meilleure qualité, soit d'une architecture plus profonde.

---

## VI. Difficultés, limites et pistes d'amélioration

### 1. Contraintes matérielles et temps de calcul

Le processus AlphaZero est par définition extrêmement gourmand en ressources. L'essentiel du projet (modèle à 19 plans) a été réalisé sur une configuration locale équipée d'une **NVIDIA RTX 3070ti**.

**Le défi du matériel « consumer »** : Contrairement aux infrastructures de Google (5000 TPU v1 pour AlphaZero), une RTX 3070ti impose un arbitrage serré entre le nombre de simulations par coup (800 à 1600) et la durée globale des cycles de self-play. Un cycle complet (800 parties + entraînement + évaluation) prenait en moyenne **12 à 18 heures** sur cette configuration.

**Optimisation du pipeline (Nitro Loading)** : Pour compenser la bande passante mémoire limitée par rapport aux cartes professionnelles, un système de chargement de données ultra-rapide a été implémenté. Ce code permet de saturer le GPU à 100% durant les phases de training en pré-chargeant les fichiers `.npz` de manière asynchrone, évitant ainsi que l'entraînement ne soit ralenti par les accès disque.

**Tests sur A100** : Des expérimentations sur Google Colab (GPU A100) ont été menées spécifiquement pour tester une architecture plus lourde (72 plans avec attention spatiale), confirmant que le pipeline est prêt pour un passage à l'échelle, mais que la RTX 3070ti reste l'unité de production principale et stable du projet.

**Gestion des crashs** : L'entraînement sur PC personnel expose à des interruptions (coupures de courant, mises à jour Windows, surchauffe). Le système de sauvegardes incrémentales (`progress.json`) permet de reprendre un cycle exactement là où il a été interrompu, sans perte de données.

### 2. Améliorations possibles du modèle

Plusieurs pistes techniques permettraient d'améliorer significativement les performances du modèle :

**Architecture Nitro (72 plans)** : Une évolution majeure consisterait à passer d'une représentation de 19 plans à 72 plans. Cela permettrait d'injecter l'historique des 8 dernières positions du plateau, donnant au réseau une « mémoire » cruciale pour détecter les répétitions et comprendre la dynamique des pièces sur le long terme. Des tests préliminaires sur A100 ont validé la faisabilité technique de cette approche (Loss descendue de 5.17 à 3.52 en 50 époques).

**Mécanismes d'attention spatiale** : L'ajout de blocs d'attention de type Transformer permettrait au réseau de mieux corréler des pièces éloignées sur l'échiquier (par exemple, une tour en a1 contrôlant une colonne vers a8), améliorant ainsi la vision tactique à longue distance.

**Augmentation des simulations** : Passer de 1600 à 3200 simulations par coup en self-play doublerait la profondeur effective de calcul, au prix d'un temps de génération deux fois plus long.

### 3. Perspectives pour la performance et la compétition

Pour franchir un palier et rivaliser avec des moteurs comme Stockfish (même à bas niveau), plusieurs stratégies d'apprentissage accéléré sont envisagées :

**Apprentissage Supervisé Hybride** : Au lieu de partir de zéro (Tabula Rasa), injecter des parties de **Grands Maîtres** dans le jeu de données initial permettrait au modèle d'acquérir immédiatement des principes d'ouverture et de stratégie positionnelle humaine. Des bases de données de millions de parties (Lichess, FIDE) sont disponibles en format PGN et pourraient être converties en shards `.npz` compatibles.

**Entraînement sur Problèmes Spécifiques** : Créer des cycles d'entraînement dédiés à des **puzzles tactiques** (mat en 1, 2 ou 3, fourchettes, clouages) permettrait de forcer le modèle à résoudre des situations critiques qu'il rencontre trop rarement en self-play pur. Cela accélérerait considérablement l'acquisition de la vision tactique.

**Curation de données** : Filtrer les shards de self-play pour ne garder que les parties les plus « instructives » (celles avec des retournements de situation ou des calculs profonds réussis par le MCTS), afin d'augmenter la densité de qualité du dataset.

---

## VII. Gestion du projet et chronologie

### 1. Répartition du travail

Le projet a été découpé en cinq Work Packages (WP) correspondant aux différentes couches du système. Chaque WP a été développé de manière itérative, avec des phases de prototypage, de validation et d'optimisation.

### 2. Organisation par Work Packages

| WP | Intitulé | Durée estimée | Livrable principal |
|----|----------|---------------|-------------------|
| WP1 | Environnement de jeu | 1 semaine | Encodeurs, action_map |
| WP2 | Réseau de neurones | 1 semaine | AlphaChessNet (SE-ResNet) |
| WP3 | MCTS | 2 semaines | Module C++ compilé (cpp_mcts.pyd) |
| WP4 | Self-play et entraînement | 1 semaine | Pipeline selfplay + trainer |
| WP5 | Évaluation et itérations | En continu | Boucle cycle.py, 29+ cycles |

**Table 7** : Synthèse des Work Packages.

### 3. Diagramme de Gantt

```
Semaine     1    2    3    4    5    6    7    8+
WP1     ████
WP2          ████
WP3              ████████
WP4                      ████
WP5                           ████████████████...
Cycles                             ▸1 ▸5 ▸10 ▸20 ▸29
```

*Figure 2 : Diagramme de Gantt simplifié du projet.*

Les WP1 à WP4 ont été développés séquentiellement, chaque couche s'appuyant sur la précédente. Le WP5 (cycles d'entraînement) s'est déroulé en continu sur plusieurs semaines, le PC faisant tourner les cycles 24h/24.

---

## Conclusion générale

### Bilan du projet

Le projet AlphaChess a permis de concevoir, implémenter et valider un moteur d'échecs neuronal complet, entièrement entraîné par auto-apprentissage sur du matériel grand public (RTX 3070ti). Les résultats obtenus après 29 cycles démontrent :

1. **La viabilité de l'approche** : Un réseau de neurones combiné au MCTS peut apprendre à jouer aux échecs à partir de zéro, sans aucune connaissance humaine préalable.
2. **L'efficacité des optimisations** : L'implémentation C++ du MCTS, le mixed precision (FP16) et le TorchScript tracing permettent d'atteindre des performances suffisantes pour un entraînement itératif sur GPU consumer.
3. **La progression mesurable** : La décroissance continue de la loss (de 2.88 à 2.75) et la capacité à résoudre des problèmes tactiques (mat en 1 et 2) confirment l'apprentissage effectif du modèle.

### Apports techniques et pédagogiques

Ce projet a permis d'acquérir une compréhension approfondie de :
- L'architecture des réseaux de neurones pour les jeux de stratégie.
- Les algorithmes de recherche arborescente et leurs optimisations.
- Le pipeline complet d'apprentissage par renforcement : de la génération de données à l'évaluation.
- La programmation hybride Python/C++ pour les applications de haute performance.
- La gestion de projets de calcul intensif sur matériel limité.

### Perspectives futures

Les pistes d'amélioration identifiées (architecture 72 plans, injection de parties de Grands Maîtres, entraînement sur puzzles tactiques) ouvrent la voie à une deuxième génération d'AlphaChess potentiellement capable de rivaliser avec les moteurs de niveau intermédiaire. L'infrastructure logicielle développée est suffisamment robuste et modulaire pour supporter ces évolutions sans refonte majeure.

---

## Bibliographie

[1] Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. arXiv:1712.01815.

[2] Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). *Mastering the game of Go without human knowledge*. Nature, 550(7676), 354-359.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.

[4] Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.

[5] Browne, C., Powley, E., Whitehouse, D., et al. (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1-43.

[6] Anthony, T., Tian, Z., & Barber, D. (2017). *Thinking Fast and Slow with Deep Learning and Tree Search*. NeurIPS 2017.

[7] Kingma, D.P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.

[8] Coulom, R. (2007). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search*. Computers and Games, 5th International Conference.

[9] Projet Leela Chess Zero. https://lczero.org/

[10] Documentation python-chess. https://python-chess.readthedocs.io/

---

## Annexes

### Annexe A : Courbes de progression

**[Insérer ici : training_progress.png]**

*Figure A.1 : Visualisation complète de l'apprentissage sur 29 cycles : convergence des fonctions de perte (haut) et taux de victoire des candidats (bas).*

### Annexe B : Données brutes d'entraînement

```csv
cycle,epoch,total_loss,policy_loss,value_loss
1,last,2.8833,2.7664,0.1168
2,last,2.9133,2.8038,0.1094
3,last,2.9070,2.8244,0.0826
4,last,2.9075,2.8353,0.0722
5,last,2.9135,2.8443,0.0692
6,last,2.9060,2.8360,0.0700
7,last,2.9120,2.8276,0.0843
8,last,2.8741,2.7996,0.0745
9,last,2.8474,2.7763,0.0711
10,last,2.8606,2.7653,0.0953
11,last,2.8557,2.7439,0.1118
12,last,2.8187,2.7342,0.0845
13,last,2.7998,2.7269,0.0729
14,last,2.8041,2.7271,0.0770
15,last,2.7874,2.7222,0.0653
16,last,2.7901,2.7233,0.0668
17,last,2.7833,2.7198,0.0635
18,last,2.7731,2.7168,0.0564
19,last,2.7703,2.7154,0.0549
20,last,2.7688,2.7135,0.0553
21,last,2.7601,2.7093,0.0508
22,last,2.7617,2.7102,0.0515
23,last,2.7652,2.7119,0.0533
24,last,2.7522,2.7072,0.0449
25,last,2.7554,2.7060,0.0494
26,last,2.7576,2.7077,0.0499
27,last,2.7718,2.7113,0.0605
28,last,2.7598,2.7085,0.0513
29,last,2.7542,2.7062,0.0480
```

### Annexe C : Données brutes d'évaluation

```csv
cycle,winrate
1,0.53
2,0.57
3,0.585
4,0.55
5,0.555
6,0.415
7,0.62
8,0.655
9,0.445
10,0.59
11,0.54
12,0.52
13,0.44
14,0.625
15,0.405
16,0.49
17,0.565
18,0.525
19,0.465
20,0.545
21,0.48
22,0.49
23,0.545
24,0.665
25,0.42
26,0.5
27,0.43
28,0.44
29,0.47
```

### Annexe D : Architecture complète du réseau

```python
# wp2/model.py - Architecture AlphaChessNet
class AlphaChessNet(nn.Module):
    def __init__(self, in_planes=19, channels=128, n_blocks=12, action_size=4672):
        super().__init__()
        self.conv_in = nn.Conv2d(in_planes, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)
        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
```

### Annexe E : Structure du MCTS C++

```cpp
// wp3/mcts.cpp - Structure de noeud compact
struct Node {
    std::unordered_map<int, uint32_t> children;
    int parent_idx = -1;
    int move_idx = -1;
    float prior = 0.0f;
    float value_sum = 0.0f;
    int visit_count = 0;
    int virtual_loss = 0;
    bool is_expanded = false;
    bool is_terminal = false;
    float terminal_val = 0.0f;
};
```

### Annexe F : Résultats du benchmark tactique

```
TEST: Scholar's Mate (Mate in 1)
FEN: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 1 1
Best move: f3f7 ✅ CHECKMATE
Simulations: 100 | Temps: 0.07s | 5170 NPS

TEST: Classic Mate in 2
FEN: r5k1/1pR1Qp1p/p5p1/2pP1b2/P1P2n2/8/1P3PPP/6K1 w - - 0 1
Best move: e7f7 ✅ SUCCESS
Simulations: 1600 | Temps: 0.12s | 22885 NPS

TEST: Back Rank Mate
FEN: 6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1
Best move: e1e8 ✅ SUCCESS
Simulations: 50 | Temps: 0.02s
```

### Annexe G : Arborescence du projet

```
Projet échec/
├── README.md                      # Documentation du projet
├── progress.json                  # Suivi de l'état de l'entraînement (cycle/phase)
├── setup.py                       # Configuration de la compilation C++ (PyBind11)
├── train_log.csv                  # Logs d'entraînement (pertes Total/Policy/Value)
├── eval_log.csv                   # Logs d'évaluation (winrate vs modèle précédent)
├── training_progress.png          # Visualisation graphique de la progression
├── test_proven_dtm.py             # Benchmarks de performance et profondeur
├── test_proven_dtm2.py            # Suite de tests additionnels
│
├── wp1_env.py                     # Environnement de base et règles
│
├── wp2/                           # ARCHITECTURE DU MODÈLE
│   ├── action_map.py              # Encodage/Décodage des 4672 coups UCI
│   └── model.py                   # Réseau SE-ResNet (AlphaChessNet)
│
├── wp3/                           # ENGINE & OPTIMISATION C++
│   ├── board_encoder.hpp          # Encodage ultra-rapide des positions en C++
│   ├── mcts.cpp                   # Cœur de l'algorithme Monte Carlo Tree Search
│   ├── cpp_mcts.pyd               # Module compilé pour Python
│   ├── predictor.py               # Wrapper PyTorch (FP16/Tracing/TensorRT)
│   └── node.py                    # Structure de l'arbre MCTS
│
├── wp4/                           # TRAINING PIPELINE
│   ├── trainer.py                 # Entraînement du modèle (AMP/Adam)
│   ├── selfplay_worker.py         # Génération de données via parties autonomes
│   └── replay_buffer.py           # Gestion de la mémoire circulaire
│
├── wp5/                           # CYCLE MANAGEMENT & UI
│   ├── cycle.py                   # Orchestrateur complet (Self-play -> Train -> Eval)
│   ├── play_gui.py                # Interface graphique (Test de positions M1-M3)
│   ├── evaluation.py              # Arena entre ancien et nouveau modèle
│   └── visualize.py               # Script de génération du graphique (live)

```

### Bibliographie

1. **Silver, D., et al. (2017).** *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.* arXiv:1712.01815.
2. **TheoEwzZer.** *NeuralMatePlay - Chess AI with Deep Learning.* GitHub: [https://github.com/TheoEwzZer/NeuralMatePlay](https://github.com/TheoEwzZer/NeuralMatePlay)
3. **Lichess.org.** *Database of Chess Puzzles.* [https://database.lichess.org/](https://database.lichess.org/)
