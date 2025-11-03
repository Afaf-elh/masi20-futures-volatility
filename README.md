# Prévision de la Volatilité des Contrats Futures sur MASI20 - Approche Hybride

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Aperçu du Projet

Cette étude propose une **approche hybride** pour la prévision de la volatilité des contrats à terme sur l'indice **MASI20** de la **Bourse de Casablanca**. En combinant modèles économétriques traditionnels (GARCH) et techniques avancées d'apprentissage automatique (LSTM, XGBoost), ce travail fournit un cadre prédictif adapté aux spécificités du marché marocain, tout en offrant une analyse comparative avec d'autres marchés émergents et frontières.

### 🌍 Marchés Étudiés

**Marchés Émergents :**
- **Inde** (Nifty 50) : 2015-01-01 à 2024-12-31
- **Afrique du Sud** (JSE Top 40) : 2015-01-01 à 2024-12-31

**Marchés Frontières :**
- **Maroc** (MASI20) : 2020-01-01 à 2024-12-31
- **Vietnam** (VN30) : 2017-01-01 à 2024-12-31

*Justification : Les périodes d'étude reflètent la disponibilité des données liée au lancement des marchés à terme. Le marché marocain des futures sur le MASI20 étant récent, les données commencent en 2020.*

## 🎯 Objectifs

- Préparer et harmoniser les données financières pour 4 marchés (Maroc, Vietnam, Afrique du Sud, Inde)
- Analyser la volatilité via modèles économétriques (GARCH, EGARCH, GJR-GARCH)
- Simuler les prix de futures (Cost of Carry, Convenience Yield, Monte Carlo)
- Comparer les dynamiques de volatilité entre marchés (corrélations, contagion, clustering)
- Développer des modèles de prévision hybrides combinant économétrie et apprentissage automatique
- Évaluer la performance des modèles (RMSE, R², intervalles de confiance)

## 📊 Résultats Clés

### 🏆 Performance des Modèles

| **Pays** | **LSTM (RMSE)** | **LightGBM (RMSE)** | **Random Forest (RMSE)** |
|----------|-----------------|---------------------|--------------------------|
| **Maroc** | **0.1089** | 0.8416 | 0.8014 |
| **Vietnam** | **0.1277** | 1.3434 | 1.3690 |
| **Afrique du Sud** | **0.1123** | 0.7352 | 0.7492 |
| **Inde** | **0.1313** | 0.8005 | 0.7997 |

**Contextualisation RMSE :** Avec une volatilité quotidienne généralement entre 0,5% et 3%, un RMSE de **0,1089** pour le Maroc indique une erreur moyenne de seulement **~0,11 point de pourcentage**, démontrant une précision exceptionnelle des modèles LSTM.

### 📈 Principales Conclusions

- **Supériorité des LSTM** : Dominance dans tous les pays avec RMSE très bas (0.1089-0.1313) et R² élevés (87.46%-95.78%)
- **Meilleur modèle GARCH** : EGARCH avec distribution t (EGARCH-t-21) dans tous les pays
- **Approche hybride** : Performances globales supérieures en combinant économétrie et IA
- **Similitudes structurelles** entre Maroc et Vietnam (marchés frontières)
- **Différences notables** avec marchés émergents (Inde, Afrique du Sud) liées à la liquidité et profondeur de marché

## 🗂️ Données

Données quotidiennes 2015-2024 provenant de sources financières reconnues (Yahoo Finance, TradingView, Trading Economics) :

- **Prix des indices** et contrats futures (quand disponibles)
- **Variables financières** : taux sans risque, dividendes, volumes
- **Prétraitement** : Standardisation, rééchantillonnage quotidien, gestion valeurs manquantes

### Cas Spécial Maroc (MASI20)
Les données futures sont calculées analytiquement via le modèle **Cost of Carry** :F = S × e^( (r - q) × T )
avec calcul des échéances selon règles AMMC (3ème vendredi mars, juin, septembre, décembre)

## 🧮 Méthodologie

### 📐 Économétrie Financière
- **GARCH/EGARCH/GJR-GARCH** avec sélection automatique des paramètres (BIC)
- Tests de stationnarité et validation (Durbin-Watson)
- Distributions : Normale, Student-t, GED

### 🤖 Apprentissage Automatique
- **Random Forest, XGBoost, LightGBM** pour comparaison
- **LSTM** pour capture des dépendances temporelles longues
- **Modèles hybrides** combinant résidus GARCH et prédictions IA

### 📊 Analyse Comparative
- **Corrélations dynamiques** (fenêtre 60 jours)
- **Analyse de contagion** (seuil 95ème percentile)
- **Clustering** par distance euclidienne

## 🗂️ Structure du Dépôt

```plaintext
├── notebooks/
│   ├── 1_exploration_donnees_ameliore.py          # Exploration et rapport des données
│   ├── 2_harmonisation_donnees_ameliore.py        # Harmonisation et fusion des jeux de données
│   ├── 3_calcul_volatilite_simulation_futures.py  # Calcul de la volatilité et simulation des futures
│   ├── 4_analyse_comparative_ameliore.py          # Analyses inter‑pays (corrélations, contagion)
│   ├── 5_modeles_prediction_avances_ameliore.py   # Entraînement des modèles IA et hybrides
│   ├── 6_visualisations_erreur_ameliore.py        # Visualisations des erreurs et métriques
│   ├── utils.py                                   # Fonctions utilitaires (chargement, stats, tracés)
│   └── config.py                                  # Paramètres globaux (chemins, hyperparamètres)
├── data/                                          # Données brutes (non incluses pour raisons de taille)
└── ...                                            # Autres dossiers créés à l’exécution (volatilite, analyse_comparative, modeles_prediction)
```

⚙️ **Installation**  

1. Cloner le dépôt :

```bash
git clone https://github.com/Afaf-elh/masi20-futures-volatility.git
cd masi20-futures-volatility
```

2. Créer un environnement Python et installer les dépendances :

```bash
python3 -m venv venv
source venv/bin/activate      # Sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

  Les principaux packages utilisés sont pandas, numpy, matplotlib, plotly, arch, scikit-learn, xgboost, lightgbm, tensorflow et statsmodels.

▶️ **Exécution du pipeline**  
Le workflow complet est automatisé via le script `notebooks/main.py` qui appelle chaque étape séquentiellement. Pour exécuter une étape manuellement :

```bash
# 1. Exploration des données
python notebooks/1_exploration_donnees_ameliore.py

# 2. Harmonisation et fusion
python notebooks/2_harmonisation_donnees_ameliore.py

# 3. Calcul de la volatilité et simulation des futures
python notebooks/3_calcul_volatilite_simulation_futures.py

# 4. Analyse comparative inter‑pays
python notebooks/4_analyse_comparative_ameliore.py

# 5. Entraînement des modèles de prévision
python notebooks/5_modeles_prediction_avances_ameliore.py

# 6. Visualisation des erreurs et métriques
python notebooks/6_visualisations_erreur_ameliore.py
```  

  Les résultats (CSV, images PNG et HTML) seront générés dans les dossiers configurés (data_harmonisee/, volatilite/, analyse_comparative/, modeles_prediction/, rapport_final/visualisations/, etc.). Notez que pour des raisons de taille, ces sorties ne sont pas versionnées dans le dépôt ; elles sont créées lors de l’exécution.

🔍 **Principaux résultats**

* Les performances des modèles mettent en évidence la supériorité de l'approche hybride pour le marché marocain :

* Supériorité des LSTM : RMSE très bas (0.1089 au Maroc, 0.1277 au Vietnam), R² élevé (87.46%-95.78%) ; excellents pour dépendances temporelles, surpassant LightGBM et Random Forest (RMSE ~0.8-1.3).

* Modèles GARCH : EGARCH-t-21 domine avec AIC entre 2358.76 et 8691.62 ; efficace pour asymétrie et queues épaisses.

* Ensembles hybrides : R² >95%, RMSE 0.28-0.61 ; réduction de variance par combinaison.

* Intervalles de confiance : Couverture 92-98%, indiquant une bonne estimation de l'incertitude pour trading prudent.

👩‍💻 **Auteure et encadrement**  
  Ce projet a été réalisé par Afafe El Hilali dans le cadre d’un mémoire de Master « Finance, Actuariat & Data Science » à l’Université Abdelmalek Essaâdi (2025). L’encadrement scientifique a été assuré par Pr. Mouad El Kharrim en collaboration avec la Bourse de Casablanca.

📄 **Licence**  
  Le code est diffusé sous licence MIT. Vous êtes libre de l’utiliser, le modifier et le distribuer en citant l’auteure.

