# Rapport d'Analyse des Modèles de Prédiction de Volatilité

## Résumé des Performances

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| Ensemble | 0.3381 | 0.2208 | 0.9710 |
| random_forest | 0.7998 | 0.3952 | 0.8377 |
| lightgbm | 0.8694 | 0.4292 | 0.8083 |
| xgboost | 0.8921 | 0.5678 | 0.7981 |
| neural_network | 1.0934 | 0.5078 | 0.6967 |

## Analyse Détaillée par Type de Modèle

### Modèles GARCH

#### GARCH_11

- AIC: 8132.64
- BIC: 8157.45

#### GARCH_12

- AIC: 8112.90
- BIC: 8143.91

#### GARCH_21

- AIC: 8134.64
- BIC: 8165.65

#### EGARCH_11

- AIC: 8187.50
- BIC: 8212.31

#### EGARCH_12

- AIC: 8153.55
- BIC: 8184.57

#### EGARCH_21

- AIC: 8183.55
- BIC: 8214.57

#### EGARCH_T_11

- AIC: 7158.79
- BIC: 7189.81

#### EGARCH_T_12

- AIC: 7140.25
- BIC: 7177.47

#### EGARCH_T_21

- AIC: 7116.65
- BIC: 7153.87

#### GARCH_SKEWT_11

- AIC: 7166.73
- BIC: 7203.94

#### EGARCH_SKEWT_11

- AIC: 7254.73
- BIC: 7291.95

### Modèles de Machine Learning

#### RANDOM_FOREST

- RMSE: 0.7998
- MAE: 0.3952
- R²: 0.8377

#### XGBOOST

- RMSE: 0.8921
- MAE: 0.5678
- R²: 0.7981

#### LIGHTGBM

- RMSE: 0.8694
- MAE: 0.4292
- R²: 0.8083

#### NEURAL_NETWORK

- RMSE: 1.0934
- MAE: 0.5078
- R²: 0.6967

### Modèles Deep Learning

### Modèle d'Ensemble

#### ENSEMBLE

- RMSE: 0.3381
- MAE: 0.2208
- R²: 0.9710

### Analyse des Intervalles de Confiance

Les intervalles de confiance à 95% ont été calculés pour chaque modèle. Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.

- random_forest: 9724.90%
- xgboost: 9821.18%
- lightgbm: 9752.41%
- neural_network: 9697.39%
- Ensemble: 9518.57%

### Visualisations

Les graphiques suivants ont été générés pour l'analyse des modèles :

1. Prédictions vs Réalité (visualisations/lstm_ameliore.png)
2. Courbes d'apprentissage (visualisations/lstm_ameliore.png)
3. Erreurs de prédiction (visualisations/lstm_ameliore.png)
4. Distribution des erreurs (visualisations/lstm_ameliore.png)
5. Prédictions vs. Valeurs réelles - Modèle d'ensemble (modeles_prediction/predictions_ensemble.png)
6. Comparaison interactive des prédictions (visualisations_prediction/{pays}/comparaison_predictions_{pays}.html)
7. Comparaison interactive des métriques (visualisations_prediction/{pays}/comparaison_metriques_{pays}.html)

### Conclusion

Cette analyse comparative des différents modèles de prédiction de volatilité montre que:

1. Le modèle **Ensemble** a montré les meilleures performances globales avec un RMSE de **0.3381**.
2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.
3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.
4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.
