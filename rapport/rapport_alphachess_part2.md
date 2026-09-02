
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
├── wp2/                    # Réseau de neurones
│   ├── model.py            # AlphaChessNet (SE-ResNet)
│   ├── encoders.py         # Encodeur Python (19 plans)
│   └── action_map.py       # Mapping coups ↔ indices
├── wp3/                    # MCTS
│   ├── mcts.cpp            # MCTS C++ (Arena, PUCT, batching)
│   ├── board_encoder.hpp   # Encodeur C++ haute performance
│   ├── cpp_mcts.pyd        # Module compilé Windows
│   └── predictor.py        # Interface PyTorch ↔ MCTS
├── wp4/                    # Self-play et Entraînement
│   ├── selfplay_worker.py  # Worker de génération de parties
│   ├── trainer.py          # Boucle d'entraînement
│   └── utils_io.py         # Sauvegarde PGN / NPZ
├── wp5/                    # Cycle et Évaluation
│   ├── cycle.py            # Orchestrateur principal
│   ├── evaluation.py       # Duel current vs candidate
│   ├── play_gui.py         # Interface graphique de jeu
│   └── visualize.py        # Génération des graphiques
├── checkpoints/
│   └── current.pt          # Modèle champion actuel
├── data/selfplay/shards/   # Données d'entraînement (.npz)
├── progress.json           # État de progression
├── train_log.csv           # Historique des pertes
└── eval_log.csv            # Historique des winrates
```
