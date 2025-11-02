"""
Module de configuration pour le projet de prédiction de volatilité des futures sur MASI20.
Ce module centralise les paramètres de configuration utilisés par les différents scripts du projet.
"""

import os
from datetime import datetime

# Paramètres généraux du projet
PROJET = {
    'titre': 'Prédiction de la volatilité des futures sur MASI20',
    'periode_debut': '2015-01-01',
    'periode_fin': '2024-12-31',
    'version': '2.1.0',
    'date_mise_a_jour': datetime.now().strftime('%Y-%m-%d')
}

# Chemins des répertoires
CHEMINS = {
    'data': 'data',
    'data_harmonisee': 'data_harmonisee',
    'volatilite': 'volatilite',
    'analyse_comparative': 'analyse_comparative',
    'modeles_prediction': 'modeles_prediction',
    'rapport_final': 'rapport_final',
    'visualisations': os.path.join('rapport_final', 'visualisations'),
    'resultats': 'resultats',
    'tableaux_bord': 'tableaux_bord',  # Répertoire pour les tableaux de bord
    'logs': 'logs',  # Nouveau répertoire pour les fichiers de log
    'visualisations_prediction': 'visualisations/prediction',
    'rapports_prediction': 'rapports/prediction'
}

# Créer les répertoires s'ils n'existent pas
for chemin in CHEMINS.values():
    os.makedirs(chemin, exist_ok=True)

# Pays étudiés - Standardisation des noms selon les instructions
PAYS = {
    'maroc': {
        'nom': 'Maroc',
        'indice': 'Masi20',
        'dossier': 'Maroc',
        'couleur': '#c1272d',
        'periode_debut': '2020-01-01',  # Période spécifique pour le Maroc
        'periode_fin': '2025-01-01'
    },
    'vietnam': {
        'nom': 'Vietnam',
        'indice': 'VN30',
        'dossier': 'Vietnam',
        'couleur': '#ffcd00',
        'periode_debut': '2017-01-01',  # Période spécifique pour le Vietnam
        'periode_fin': '2025-01-01'
    },
    
    'afrique_sud': {
        'nom': 'Afrique',  # Conservé comme "Afrique" selon les instructions
        'indice': 'JSE40',
        'dossier': 'Afrique',
        'couleur': '#007a4d',
        'periode_debut': '2015-01-01',
        'periode_fin': '2025-01-01'
    },
    'inde': {
        'nom': 'India',  # Conservé comme "India" selon les instructions
        'indice': 'Nifty50',
        'dossier': 'India',
        'couleur': '#ff9933',
        'periode_debut': '2015-01-01',
        'periode_fin': '2025-01-01'
    }
}

# Pour chaque pays, créer les sous-répertoires nécessaires
for pays_info in PAYS.values():
    dossier = pays_info['dossier']
    for chemin_base in ['data_harmonisee', 'volatilite', 'analyse_comparative', 'modeles_prediction']:
        os.makedirs(os.path.join(CHEMINS[chemin_base], dossier), exist_ok=True)


# Paramètres des modèles de volatilité
MODELES_VOLATILITE = {
    'historique': {
        'nom': 'Volatilité historique',
        'description': 'Écart-type mobile des rendements',
        'fenetre': 30,
        'annualisation': True
    },
    'garch': {
        'nom': 'GARCH',
        'description': 'Generalized Autoregressive Conditional Heteroskedasticity',
        'p': 1,
        'q': 1,
        'optimisation': True,
        'annualisation': True,
        'p_max': 5,
        'q_max': 5,
        'top_k': 5
    },
    'egarch': {
        'nom': 'EGARCH',
        'description': 'Exponential GARCH',
        'p': 1,
        'q': 1,
        'optimisation': True,
        'annualisation': True,
        'p_max': 5,
        'q_max': 5
    },
    'gjr_garch': {
        'nom': 'GJR-GARCH',
        'description': 'Glosten-Jagannathan-Runkle GARCH',
        'p': 1,
        'q': 1,
        'optimisation': True,
        'annualisation': True,
        'p_max': 5,
        'q_max': 5
    }
}

# Paramètres des modèles de simulation des futures
SIMULATION_FUTURES = {
    'cost_of_carry': {
        'nom': 'Cost of Carry',
        'description': 'Modèle de base pour la simulation des futures',
        'taux_sans_risque': {
            'maroc': 0.03,
            'vietnam': 0.04,
            'afrique_sud': 0.05,
            'inde': 0.04
        },
        'maturite': 90  # en jours
    },
    'monte_carlo': {
        'nom': 'Monte Carlo',
        'description': 'Simulation Monte Carlo avec volatilité stochastique',
        'n_simulations': 5000,
        'taux_sans_risque': {
            'maroc': 0.03,
            'vietnam': 0.04,
            'afrique_sud': 0.05,
            'inde': 0.04
        },
        'maturite': 90,  # en jours
        'vol_of_vol': 0.2,
        'mean_reversion': 0.1
    },
    'convenience_yield': {
        'nom': 'Convenience Yield',
        'description': 'Modèle intégrant le rendement de convenance',
        'taux_sans_risque': {
            'maroc': 0.03,
            'vietnam': 0.04,
            'afrique_sud': 0.05,
            'inde': 0.04
        },
        'storage_cost': 0.01,
        'convenience_yield': {
            'maroc': 0.02,
            'vietnam': 0.02,
            'afrique_sud': 0.02,
            'inde': 0.02
        },
        'maturite': 90  # en jours
    }
}

# Paramètres des modèles de machine learning avec analyse de sensibilité
MODELES_ML = {
    'xgboost': {
        'nom': 'XGBoost',
        'description': 'eXtreme Gradient Boosting',
        'parametres': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        },
        'optimisation': True,
        'analyse_sensibilite': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    },
    'lstm': {
        'nom': 'LSTM',
        'description': 'Long Short-Term Memory',
        'parametres': {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'activation': 'relu',
            'epochs': 20,
            'batch_size': 32,
            'patience': 10
        },
        'optimisation': False,
        'analyse_sensibilite': {
            'units': [32, 50, 64, 128],
            'dropout': [0.1, 0.2, 0.3],
            'recurrent_dropout': [0.1, 0.2, 0.3],
            'activation': ['relu', 'tanh'],
            'batch_size': [16, 32, 64]
        }
    },
    'cnn_lstm': {
        'nom': 'CNN-LSTM',
        'description': 'Convolutional Neural Network + LSTM',
        'parametres': {
            'filters': 64,
            'kernel_size': 3,
            'lstm_units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'activation': 'relu',
            'epochs': 20,
            'batch_size': 32,
            'patience': 10
        },
        'optimisation': False,
        'analyse_sensibilite': {
            'filters': [32, 64, 128],
            'kernel_size': [2, 3, 5],
            'lstm_units': [32, 50, 64],
            'dropout': [0.1, 0.2, 0.3],
            'activation': ['relu', 'tanh']
        }
    },
    'random_forest': {
        'nom': 'Random Forest',
        'description': 'Random Forest Regressor',
        'parametres': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        },
        'optimisation': True,
        'analyse_sensibilite': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'svr': {
        'nom': 'SVR',
        'description': 'Support Vector Regression',
        'parametres': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        },
        'optimisation': True,
        'analyse_sensibilite': {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    },
    'ensemble': {
        'nom': 'Ensemble',
        'description': 'Modèle ensembliste combinant plusieurs approches',
        'modeles_base': ['xgboost', 'random_forest', 'svr'],
        'modele_final': 'ridge',
        'parametres': {
            'cv': 5
        },
        'optimisation': False,
        'analyse_sensibilite': {
            'modeles_base': [
                ['xgboost', 'random_forest'],
                ['xgboost', 'svr'],
                ['random_forest', 'svr'],
                ['xgboost', 'random_forest', 'svr']
            ],
            'modele_final': ['ridge', 'lasso', 'elasticnet']
        }
    },
    'lightgbm': {  # Ajout du modèle LightGBM
        'nom': 'LightGBM',
        'description': 'Light Gradient Boosting Machine',
        'parametres': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'optimisation': True,
        'analyse_sensibilite': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'num_leaves': [15, 31, 63],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    },
    'garch_ml': {  # Ajout du modèle hybride GARCH-ML
        'nom': 'GARCH-ML',
        'description': 'Modèle hybride combinant GARCH et Machine Learning',
        'parametres': {
            'p': 1,
            'q': 1,
            'model': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'optimisation': True
    },
    'regime_switching': {  # Ajout du modèle à changement de régime
        'nom': 'Regime Switching',
        'description': 'Modèle à changement de régime pour la volatilité',
        'parametres': {
            'k_regimes': 2,
            'order': 1,
            'switching_variance': True
        },
        'optimisation': False
    }
}


# Paramètres pour l'analyse comparative
ANALYSE_COMPARATIVE = {
    'correlation_dynamique': {
        'nom': 'Corrélation conditionnelle dynamique',
        'description': 'Analyse de la corrélation dynamique entre les volatilités des différents pays'
    },
    'contagion': {
        'nom': 'Analyse de contagion',
        'description': 'Analyse de la contagion entre marchés en période de forte volatilité',
        'seuil_percentile': 95
    },
    'similarite': {
        'nom': 'Analyse de similarité',
        'description': 'Analyse de la similarité entre les marchés',
        'methode': 'euclidean',
        'linkage': 'ward'
    }
}

# Paramètres pour le feature engineering
FEATURE_ENGINEERING = {
    'fenetres_moyennes_mobiles': [5, 10, 22, 66],  # Jours, semaines, mois, trimestres
    'lags': list(range(1, 6)),
    'indicateurs_techniques': ['rsi', 'bollinger_width'],
    'features_temporelles': ['jour_semaine', 'mois', 'trimestre']
}

# Paramètres pour l'évaluation des modèles
EVALUATION = {
    'metriques': ['mse', 'rmse', 'mae', 'r2', 'mape'],
    'validation_croisee': {
        'methode': 'time_series_split',
        'n_splits': 5
    },
    'horizon_prediction': [1, 5, 22],  # Jour, semaine, mois
    'importance_variables': True,  # Ajout de l'analyse d'importance des variables
    'intervalles_confiance': True,  # Ajout des intervalles de confiance
    'niveau_confiance': 0.95  # Niveau de confiance pour les intervalles
}

# Paramètres de validation croisée temporelle
VALIDATION_CROISEE = {
    'methode': 'time_series_split',  # Validation croisée temporelle
    'n_splits': 5,
    'test_size': 0.2,
    'random_state': 42,
    'gap': 0  # Ajout d'un paramètre pour le gap entre train et test
}

# Paramètres de métriques d'évaluation
METRIQUES = {
    'regression': ['mse', 'rmse', 'mae', 'r2', 'mape'],
    'classification': ['accuracy', 'precision', 'recall', 'f1']
}

# Paramètres de visualisation améliorés
VISUALISATION = {
    'style': 'seaborn-whitegrid',  # Style amélioré
    'palette': 'viridis',  # Palette de couleurs adaptée
    'taille_figure': (14, 8),  # Taille augmentée pour plus de lisibilité
    'dpi': 300,
    'format_sauvegarde': 'png',
    'annotations': True,  # Activer les annotations
    'legende': {
        'position': 'best',
        'taille': 10,
        'transparence': 0.8
    },
    'interactivite': {
        'activer': True,
        'bibliotheque': 'plotly',  # Utiliser plotly pour l'interactivité
        'template': 'plotly_white'
    },
    'pays_couleurs': {
        'Maroc': '#c1272d',
        'Vietnam': '#ffcd00',
        'Afrique': '#007a4d',
        'India': '#ff9933'
    },
    'evenements': {  # Ajout des événements importants pour les annotations
        'Maroc': {
            '2020-03-15': 'Début de la pandémie COVID-19',
            '2022-02-24': 'Invasion de l\'Ukraine',
            '2023-10-07': 'Conflit Israël-Hamas'
        },
        'Vietnam': {
            '2020-03-15': 'Début de la pandémie COVID-19',
            '2022-02-24': 'Invasion de l\'Ukraine'
        },
        'Afrique': {
            '2020-03-15': 'Début de la pandémie COVID-19',
            '2022-02-24': 'Invasion de l\'Ukraine'
        },
        'India': {
            '2020-03-15': 'Début de la pandémie COVID-19',
            '2022-02-24': 'Invasion de l\'Ukraine'
        }
    }
}

# Paramètres de logging
LOGGING = {
    'niveau': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'fichier': 'logs/prediction.log',
    'console': True,
    'rotation': {
        'activer': True,
        'max_bytes': 10485760,  # 10 Mo
        'backup_count': 5
    }
}

# Paramètres pour le tableau de bord
TABLEAU_BORD = {
    'titre': 'Analyse de la volatilité des futures sur indices boursiers',
    'description': 'Tableau de bord interactif pour l\'analyse comparative des marchés émergents',
    'port': 8050,
    'theme': 'plotly_white',
    'mise_a_jour_auto': False,
    'intervalle_mise_a_jour': 3600,  # en secondes (1 heure)
    'sections': [
        'Aperçu global',
        'Volatilité par pays',
        'Analyse comparative',
        'Prédictions'
    ],
    'filtres': [
        'Pays',
        'Période',
        'Modèle de volatilité',
        'Modèle de prédiction'
    ],
    'hauteur': 800,
    'largeur': 1200,
    'interactif': True,
    'boutons_zoom': [
        {'label': 'Tout', 'periode': 'all'},
        {'label': '1 an', 'periode': '1y'},
        {'label': '6 mois', 'periode': '6m'},
        {'label': '3 mois', 'periode': '3m'},
        {'label': '1 mois', 'periode': '1m'}
    ]
}

# Paramètres de détection des valeurs aberrantes
VALEURS_ABERRANTES = {
    'methode': 'zscore',  # 'zscore' ou 'iqr'
    'seuil': 3.0,
    'traitement': 'marquer',  # 'marquer', 'remplacer' ou 'supprimer'
    'analyse_separee': True  # Analyser séparément les valeurs aberrantes
}

# Paramètres des tests de stationnarité
STATIONNARITE = {
    'test': 'adf',  # 'adf' ou 'kpss'
    'seuil_pvalue': 0.05,
    'max_diff': 2,  # Nombre maximum de différenciations
    'regression': 'c',  # 'c' pour constante, 'ct' pour constante et tendance
    'autolag': 'AIC'  # Critère pour la sélection automatique du nombre de retards
}

# Paramètres pour le calcul des rendements
RENDEMENTS = {
    'methode': 'log',  # 'log' ou 'simple'
    'multiplicateur': 100  # Pour exprimer en pourcentage
}
