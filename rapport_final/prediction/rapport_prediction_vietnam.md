# Rapport d'Analyse des Modèles de Prédiction de Volatilité

## Résumé des Performances

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| lstm | 0.1277 | 0.0776 | 0.9578 |
| cnn_lstm | 0.4565 | 0.4096 | 0.4604 |
| Ensemble | 0.6089 | 0.3889 | 0.9551 |
| lightgbm | 1.3434 | 0.7571 | 0.7815 |
| random_forest | 1.3690 | 0.8171 | 0.7730 |
| xgboost | 1.7519 | 1.4084 | 0.6283 |
| neural_network | 1.8403 | 1.1445 | 0.5899 |

## Analyse Détaillée par Type de Modèle

### Modèles GARCH

#### GARCH_11

- AIC: 7468.45
- BIC: 7492.05

#### GARCH_12

- AIC: 7460.25
- BIC: 7489.76

#### GARCH_21

- AIC: 7470.45
- BIC: 7499.95

#### EGARCH_11

- AIC: 7496.91
- BIC: 7520.51

#### EGARCH_12

- AIC: 7485.51
- BIC: 7515.01

#### EGARCH_21

- AIC: 7498.68
- BIC: 7528.19

#### EGARCH_T_11

- AIC: 6447.54
- BIC: 6477.04

#### EGARCH_T_12

- AIC: 6421.26
- BIC: 6456.67

#### EGARCH_T_21

- AIC: 6391.72
- BIC: 6427.12

#### GARCH_SKEWT_11

- AIC: 6451.35
- BIC: 6486.76

#### EGARCH_SKEWT_11

- AIC: 6442.04
- BIC: 6477.44

### Modèles de Machine Learning

#### RANDOM_FOREST

- RMSE: 1.3690
- MAE: 0.8171
- R²: 0.7730

#### XGBOOST

- RMSE: 1.7519
- MAE: 1.4084
- R²: 0.6283

#### LIGHTGBM

- RMSE: 1.3434
- MAE: 0.7571
- R²: 0.7815

#### NEURAL_NETWORK

- RMSE: 1.8403
- MAE: 1.1445
- R²: 0.5899

### Modèles Deep Learning

#### LSTM

- RMSE: 0.1277
- MAE: 0.0776
- R²: 0.9578

#### CNN_LSTM

- RMSE: 0.4565
- MAE: 0.4096
- R²: 0.4604

### Modèle d'Ensemble

#### ENSEMBLE

- RMSE: 0.6089
- MAE: 0.3889
- R²: 0.9551

### Analyse des Intervalles de Confiance

Les intervalles de confiance à 95% ont été calculés pour chaque modèle. Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.

- random_forest: 9533.58%
- xgboost: 9794.78%
- lightgbm: 9608.21%
- neural_network: 9552.24%
- lstm: 9591.08%
- cnn_lstm: 9702.60%
- Ensemble: 9552.24%

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

1. Le modèle **lstm** a montré les meilleures performances globales avec un RMSE de **0.1277**.
2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.
3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.
4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.
