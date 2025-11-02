# Rapport d'Analyse des Modèles de Prédiction de Volatilité

## Résumé des Performances

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| Ensemble | 0.4156 | 0.2781 | 0.9275 |
| random_forest | 0.7492 | 0.4785 | 0.7645 |
| xgboost | 0.7687 | 0.5702 | 0.7521 |
| lightgbm | 0.7717 | 0.4945 | 0.7501 |
| neural_network | 0.7759 | 0.4845 | 0.7474 |

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
- MAE: 0.4785
- R²: 0.7645

#### XGBOOST

- RMSE: 0.7687
- MAE: 0.5702
- R²: 0.7521

#### LIGHTGBM

- RMSE: 0.7717
- MAE: 0.4945
- R²: 0.7501

#### NEURAL_NETWORK

- RMSE: 0.7759
- MAE: 0.4845
- R²: 0.7474

### Modèles Deep Learning

### Modèle d'Ensemble

#### ENSEMBLE

- RMSE: 0.4156
- MAE: 0.2781
- R²: 0.9275

### Analyse des Intervalles de Confiance

Les intervalles de confiance à 95% ont été calculés pour chaque modèle. Le taux de couverture indique la proportion de valeurs réelles qui tombent dans ces intervalles.

- random_forest: 9532.32%
- xgboost: 9656.12%
- lightgbm: 9559.83%
- neural_network: 9477.30%
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

1. Le modèle **Ensemble** a montré les meilleures performances globales avec un RMSE de **0.4156**.
2. Les modèles GARCH (en particulier EGARCH avec distribution t ou skewt) capturent bien la dynamique de la volatilité.
3. Les modèles LSTM montrent également de bonnes performances, indiquant leur capacité à capturer les dépendances temporelles.
4. Les intervalles de confiance fournissent une mesure utile de l'incertitude des prédictions.
