# Rapport d'Analyse des Modèles de Prédiction de Volatilité

## Résumé des Performances

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| lstm | 0.1089 | 0.0633 | 0.8746 |
| Ensemble | 0.2808 | 0.2012 | 0.9591 |
| cnn_lstm | 0.2847 | 0.2339 | 0.1428 |
| random_forest | 0.8014 | 0.4852 | 0.6672 |
| lightgbm | 0.8416 | 0.5149 | 0.6329 |
| neural_network | 1.0218 | 0.6532 | 0.4589 |
| xgboost | 1.0526 | 0.8926 | 0.4258 |

## Analyse Détaillée par Type de Modèle

### Modèles GARCH

#### GARCH_11

- AIC: 3337.66
- BIC: 3359.70

#### GARCH_12

- AIC: 3314.17
- BIC: 3341.72

#### GARCH_21

- AIC: 3339.66
- BIC: 3367.21

#### EGARCH_11

- AIC: 3360.21
- BIC: 3382.25

#### EGARCH_12

- AIC: 3326.81
- BIC: 3354.36

#### EGARCH_21

- AIC: 3341.71
- BIC: 3369.26

#### EGARCH_T_11

- AIC: 2458.60
- BIC: 2486.15

#### EGARCH_T_12

- AIC: 2415.42
- BIC: 2448.48

#### EGARCH_T_21

- AIC: 2358.76
- BIC: 2391.82

#### GARCH_SKEWT_11

- AIC: 2462.56
- BIC: 2495.62

#### EGARCH_SKEWT_11

- AIC: 2453.79
- BIC: 2486.85

### Modèles de Machine Learning

#### RANDOM_FOREST

- RMSE: 0.8014
- MAE: 0.4852
- R²: 0.6672

#### XGBOOST

- RMSE: 1.0526
- MAE: 0.8926
- R²: 0.4258

#### LIGHTGBM

- RMSE: 0.8416
- MAE: 0.5149
- R²: 0.6329

#### NEURAL_NETWORK

- RMSE: 1.0218
- MAE: 0.6532
- R²: 0.4589

### Modèles Deep Learning

#### LSTM

- RMSE: 0.1089
- MAE: 0.0633
- R²: 0.8746

#### CNN_LSTM

- RMSE: 0.2847
- MAE: 0.2339
- R²: 0.1428

### Modèle d'Ensemble

#### ENSEMBLE

- RMSE: 0.2808
- MAE: 0.2012
- R²: 0.9591

### Analyse des Intervalles de Confiance

Les intervalles de confiance à 95% ont été calculés pour chaque modèle. Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.

- random_forest: 9640.88%
- xgboost: 9668.51%
- lightgbm: 9613.26%
- neural_network: 9419.89%
- lstm: 9615.38%
- cnn_lstm: 9285.71%
- Ensemble: 9419.89%

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

1. Le modèle **lstm** a montré les meilleures performances globales avec un RMSE de **0.1089**.
2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.
3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.
4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.
