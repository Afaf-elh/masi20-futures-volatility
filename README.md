# Prévision de la Volatilité des Contrats Futures sur MASI20

📖 **Aperçu du projet**  
  Ce dépôt présente une étude de prévision de la volatilité des contrats à terme sur l’indice MASI20 de la Bourse de Casablanca. Le marché des dérivés marocain étant encore récent, ce projet explore des méthodes de finance quantitative, d’économétrie et d’intelligence artificielle pour proposer un cadre complet de modélisation et d’anticipation de la volatilité. L’analyse inclut également des comparaisons avec d’autres places financières (Vietnam, Afrique du Sud, Inde) pour situer le Maroc par rapport aux marchés frontières et émergents.

💡 **Objectifs**

* Préparer et harmoniser les données financières et macroéconomiques pour les marchés marocain, vietnamien, sud‑africain et indien.
* Analyser la volatilité des indices et des contrats futures via des modèles économétriques (GARCH, EGARCH, GJR‑GARCH) et des approches historiques.
* Simuler des prix de futures (méthode du coût de portage et simulation de volatilité stochastique).
* Comparer la dynamique de volatilité entre pays (corrélations dynamiques, contagion, distances).
* Construire des modèles de prévision avec des techniques de machine learning (Random Forest, XGBoost, LightGBM, MLP, LSTM, CNN‑LSTM) et des approches hybrides combinant économétrie et IA.
* Évaluer la performance des modèles (MAE, MSE, RMSE, MAPE) et analyser les erreurs de prédiction.

🗃️ **Données**  
  Les données sont quotidiennes et couvrent la période 2015–2025 (selon le pays). Elles comprennent :

* Prix de l’indice et prix du contrat future (quand disponible).
* Volumes négociés, taux sans risque, dividendes, taux de change, etc.
* Variables macroéconomiques (inflation, PIB) pour enrichir les modèles IA.
  Le rapport d’exploration (rapport_final/rapport_exploration.md) fournit un aperçu des fichiers.

### Exemple de Fichiers :

* **Masi20.csv** : 1 287 observations et six colonnes (colonne « Volume » manquante).
* **VN30.csv** : 2 568 lignes, 16 % de valeurs manquantes dans les colonnes de volume.
* **Données sud‑africaines et indiennes** : Couvre principalement 2015–2025 avec peu de valeurs manquantes.  

  L’exploration recommande d’interpoler les valeurs manquantes, d’harmoniser les formats de date et de convertir toutes les séries à une fréquence quotidienne.

🧠 **Méthodologie**  
**Économétrie**

* Modèles GARCH (GARCH, EGARCH, GJR‑GARCH) pour capturer la dépendance conditionnelle de la volatilité et l’asymétrie.
* Sélection automatique des ordres p/q via AIC/BIC et choix de la distribution des innovations (normale, Student‑t, Skew‑t, GED).
* Décomposition de la volatilité réalisée pour valider les faits stylisés.

**Machine Learning et Deep Learning**

* Modèles supervisés : Random Forest, XGBoost, LightGBM, Support Vector Regression, MLP.
* Réseaux récurrents et convolutifs : LSTM et CNN‑LSTM pour capturer les non‑linéarités et mémoires longues.
* Modèle hybride : les résidus des modèles GARCH alimentent les modèles IA, et les prédictions sont combinées via un modèle d’ensemble.

**Analyse comparative et simulation**

* Simulation de futures selon le modèle du coût de portage et par Monte‑Carlo avec volatilité stochastique.
* Comparaison inter‑pays : corrélations dynamiques, matrices de contagion (probabilités conditionnelles de volatilité élevée), distances et regroupement hiérarchique.

📁 **Structure du dépôt**

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

