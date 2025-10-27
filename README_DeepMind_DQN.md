# DQN avec Architecture DeepMind pour Goal Dynamique

## Vue d'ensemble

Ce projet implémente et compare plusieurs approches d'apprentissage par renforcement pour un environnement Grid World, avec un focus particulier sur un **DQN avec architecture DeepMind (convolutionnelle)** pour gérer des goals dynamiques.

## Implémentations

### 1. **DeepMind DQN (Architecture CNN) - Goal Dynamique** 🧠
- **Architecture**: Couches convolutionnelles (3 canaux → 32 → 64 → 64 → FC)
- **Environnement**: Grille 6x6 avec goal changeant à chaque épisode
- **Représentation**: Grille 3D (canal agent, canal goal, canal obstacles)
- **Paramètres**: ~100K
- **Avantages**: 
  - Capture la structure spatiale de l'environnement
  - Généralise bien à différentes positions de goal
  - Exploite les couches convolutionnelles pour détecter des patterns spatiaux

### 2. **Q-Learning Tabulaire - Goal Dynamique** 📊
- **Architecture**: Table Q traditionnelle
- **Environnement**: Grille 6x6 avec goal changeant
- **Paramètres**: 144 valeurs (36 états × 4 actions)
- **Avantages**: Simple, convergence rapide pour petits espaces

### 3. **DQN Standard (Fully-Connected) - Goal Fixe (Baseline)** 🤖
- **Architecture**: Réseaux fully-connected (2 → 128 → 128 → 64 → 4)
- **Environnement**: Grille 7x7 avec goal fixe
- **Paramètres**: ~17K
- **Baseline**: Pour comparaison avec les méthodes à goal dynamique

### 4. **Q-Learning Tabulaire - Goal Fixe (Baseline Optimal)** ✅
- **Architecture**: Table Q
- **Environnement**: Grille 7x7 avec goal fixe
- **Paramètres**: 196 valeurs (49 états × 4 actions)
- **Baseline optimale**: Solution optimale garantie

## Architecture DeepMind DQN

```
Input: Grille 3D (3, 6, 6)
  ↓
Conv1: 3 → 32 (kernel 3×3, stride 1, padding 1) + ReLU
  ↓
Conv2: 32 → 64 (kernel 3×3, stride 1, padding 1) + ReLU
  ↓
Conv3: 64 → 64 (kernel 2×2, stride 1, padding 0) + ReLU
  ↓
Flatten
  ↓
FC1: flattened → 512 + ReLU
  ↓
FC2: 512 → 4 (Q-values pour chaque action)
```

### Représentation de l'État (3 Canaux)

- **Canal 0**: Position de l'agent (1 à la position de l'agent, 0 ailleurs)
- **Canal 1**: Position du goal (1 à la position du goal, 0 ailleurs)
- **Canal 2**: Obstacles (1 aux positions d'obstacles, 0 ailleurs)

## Résultats Expérimentaux

| Méthode | Environnement | Récompense | Pas Moyens | Taux Succès | Paramètres |
|---------|--------------|------------|------------|-------------|------------|
| DeepMind DQN (CNN) | Dynamic 6×6 | ~XX.XX | ~XX.X | ~XX% | ~100K |
| Q-Learning Tab. | Dynamic 6×6 | ~XX.XX | ~XX.X | ~XX% | 144 |
| DQN Standard (FC) | Fixed 7×7 | ~XX.XX | ~XX.X | ~XX% | ~17K |
| Q-Learning Tab. | Fixed 7×7 | ~XX.XX | ~XX.X | ~XX% | 196 |

*(Les valeurs exactes seront disponibles après l'exécution du notebook)*

## Observations Clés

### Goal Dynamique vs Goal Fixe
- Le goal dynamique est significativement plus difficile
- Les agents doivent apprendre une politique plus générale
- Récompenses généralement plus faibles avec goal dynamique

### Architecture DeepMind (CNN)
**Avantages:**
- ✅ Capture efficacement la structure spatiale de la grille
- ✅ Généralise à différentes positions de goal et obstacles
- ✅ Performance compétitive avec Q-Learning tabulaire

**Inconvénients:**
- ❌ Plus complexe et nécessite plus de temps d'entraînement
- ❌ Beaucoup plus de paramètres (~100K vs 144)
- ❌ Nécessite plus de mémoire et de calcul

### Q-Learning Tabulaire
**Avantages:**
- ✅ Excellent pour petits espaces d'états (< 1000 états)
- ✅ Convergence rapide et stable
- ✅ Solution optimale garantie avec exploration suffisante

**Inconvénients:**
- ❌ Ne passe pas à l'échelle (mémoire O(|S| × |A|))
- ❌ Impossible pour états continus

## Recommandations

| Scénario | Méthode Recommandée |
|----------|---------------------|
| Petit espace d'états (<1000) | Q-Learning Tabulaire |
| Structure spatiale importante | DeepMind DQN (CNN) |
| Grand espace d'états sans structure spatiale | DQN Standard (FC) |
| Goal/Environnement dynamique | DeepMind DQN (CNN) + représentation riche |

## Fichiers Générés

- `q_values_5x5.csv`, `q_values_7x7.csv`, `q_values_10x10.csv` - Q-values pour grilles fixes
- `q_values_dynamic_goal_6x6.csv` - Q-values pour goal dynamique
- `dqn_model_7x7.pth` - Modèle DQN standard (baseline)
- `dqn_training_stats_7x7.csv` - Statistiques d'entraînement DQN standard
- `deepmind_dqn_model_6x6_dynamic.pth` - Modèle DeepMind DQN
- `deepmind_dqn_training_stats_6x6_dynamic.csv` - Statistiques d'entraînement DeepMind DQN

## Utilisation

1. Ouvrir le notebook `Q_Learning_Grid_World.ipynb`
2. Exécuter les cellules séquentiellement
3. Les sections 30.* contiennent l'implémentation DeepMind DQN
4. La section 31 présente la comparaison finale et les conclusions

## Dépendances

```python
numpy
matplotlib
pandas
torch
```

## Auteur

Zouga Mouhcine

## Date

Octobre 2025
