# Rapport d'Analyse des Modèles de Prédiction de Volatilité

## Résumé des Performances

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| lstm | 0.1082 | 0.0701 | 0.9335 |
| cnn_lstm | 0.3472 | 0.2869 | 0.3163 |
| Ensemble | 0.4156 | 0.2781 | 0.9275 |
| lightgbm | 0.7352 | 0.4651 | 0.7732 |
| random_forest | 0.7492 | 0.4792 | 0.7645 |
| neural_network | 0.7759 | 0.4845 | 0.7474 |
| xgboost | 1.0472 | 0.8769 | 0.5398 |

## Analyse Détaillée par Type de Modèle

### Modèles GARCH

#### GARCH_11

- AIC: 9559.00
- BIC: 9583.81

#### GARCH_12

- AIC: 9554.84
- BIC: 9585.86

#### GARCH_21

- AIC: 9561.00
- BIC: 9592.01

#### EGARCH_11

- AIC: 9619.24
- BIC: 9644.05

#### EGARCH_12

- AIC: 9611.23
- BIC: 9642.24

#### EGARCH_21

- AIC: 9617.05
- BIC: 9648.07

#### EGARCH_T_11

- AIC: 8765.60
- BIC: 8796.62

#### EGARCH_T_12

- AIC: 8746.22
- BIC: 8783.44

#### EGARCH_T_21

- AIC: 8691.62
- BIC: 8728.84

#### GARCH_SKEWT_11

- AIC: 8778.12
- BIC: 8815.34

#### EGARCH_SKEWT_11

- AIC: 8765.95
- BIC: 8803.17

### Modèles de Machine Learning

#### RANDOM_FOREST

- RMSE: 0.7492
- MAE: 0.4792
- R²: 0.7645

#### XGBOOST

- RMSE: 1.0472
- MAE: 0.8769
- R²: 0.5398

#### LIGHTGBM

- RMSE: 0.7352
- MAE: 0.4651
- R²: 0.7732

#### NEURAL_NETWORK

- RMSE: 0.7759
- MAE: 0.4845
- R²: 0.7474

### Modèles Deep Learning

#### LSTM

- RMSE: 0.1082
- MAE: 0.0701
- R²: 0.9335

#### CNN_LSTM

- RMSE: 0.3472
- MAE: 0.2869
- R²: 0.3163

### Modèle d'Ensemble

#### ENSEMBLE

- RMSE: 0.4156
- MAE: 0.2781
- R²: 0.9275

### Analyse des Intervalles de Confiance

Les intervalles de confiance à 95% ont été calculés pour chaque modèle. Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.

- random_forest: 9518.57%
- xgboost: 9656.12%
- lightgbm: 9573.59%
- neural_network: 9477.30%
- lstm: 9451.30%
- cnn_lstm: 9190.67%
- Ensemble: 9436.04%

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

1. Le modèle **lstm** a montré les meilleures performances globales avec un RMSE de **0.1082**.
2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.
3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.
4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.
